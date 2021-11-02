import time
from typing import Optional, Callable

import numpy as np

from dphox import Device, MZI, DC, Waveguide, Via, CommonLayer
from dphox.active import ThermalPS
from dphox.utils import fix_dataclass_init_docs
from pydantic.dataclasses import dataclass
from simphox.circuit import ForwardMesh
from dataclasses import asdict
import holoviews as hv
from holoviews.streams import Pipe
import panel as pn

from scipy.special import beta as beta_func
from simphox.utils import random_vector, random_unitary


def beta_pdf(x, a, b):
    return (x ** (b - 1) * (1 - x) ** (a - 1)) / beta_func(a, b)


def beta_phase(theta, a, b):
    x = np.cos(theta / 2) ** 2
    return np.abs(beta_pdf(x, a, b) * np.sin(theta / 2) * np.cos(theta / 2) / np.pi)


ps = ThermalPS(Waveguide((0.5, 5)), ps_w=2.5, via=Via((0.4, 0.4), 0.1))
dc = DC(waveguide_w=0.5, interaction_l=2.5, bend_l=1.25, interport_distance=6, gap_w=0.125)


class MeshConfig:
    arbitrary_types_allowed = True


@fix_dataclass_init_docs
@dataclass(config=MeshConfig)
class Mesh(Device):
    """Default rectangular mesh, or triangular mesh if specified
    Note: triangular meshes can self-configure, but rectangular meshes cannot.

    Attributes:
        mzi: The :code:`MZI` object, which acts as the unit cell for the mesh.
        n: The number of inputs and outputs for the mesh.
        triangular: Triangular mesh, otherwise rectangular mesh
        name: Name of the device
    """

    mesh_fn: Callable[[np.ndarray, ...], ForwardMesh]
    orig_matrix: Optional[np.ndarray] = None
    dc: DC = dc
    ps: ThermalPS = ps
    ridge: str = CommonLayer.RIDGE_SI
    name: str = 'mesh'

    def __post_init_post_parse__(self):
        self.mesh = self.mesh_fn(self.orig_matrix)
        self.interport_distance = self.dc.interport_distance
        port = {}
        self.n = self.mesh.n
        pattern_to_layer = []
        self.mzis = {}
        self.column_width = 0
        for column in self.mesh.columns:
            for node in column.nodes:
                dc_dict = asdict(self.dc)
                dc_dict['interport_distance'] *= node.bottom - node.top
                mzi = MZI(DC(**dc_dict),
                          top_internal=[self.ps.copy], top_external=[self.ps.copy],
                          bottom_internal=[self.ps.copy], bottom_external=[self.ps.copy])
                mzi.translate(node.column * mzi.size[0], node.top * self.interport_distance)
                self.mzis[(node.column, node.top, node.bottom)] = mzi.copy
                pattern_to_layer.extend(mzi.pattern_to_layer)
                self.column_width = mzi.size[0]
                if f'a{node.bottom}' not in port:
                    port[f'a{node.bottom}'] = mzi.port['a1']
                if f'a{node.top}' not in port:
                    port[f'a{node.top}'] = mzi.port['a0']
                port[f'b{node.top}'] = mzi.port['b0']
                port[f'b{node.bottom}'] = mzi.port['b1']
        self.waveguide_w = self.dc.waveguide_w
        super(Mesh, self).__init__(self.name, pattern_to_layer)
        self.port = port
        self.power_pipe = Pipe()
        self.theta_pipe = Pipe()
        self.phi_pipe = Pipe()
        self.input_pipe = Pipe()
        self.output_pipe = Pipe()
        self.status_pipe = Pipe()
        self.orig_matrix_pipe = Pipe()
        self.curr_matrix_pipe = Pipe()
        self.fidelity_matrix_pipe = Pipe()
        self.status_pipe.send('Idle')
        self.path_array, self.theta_array, self.phi_array, self.path_idx = self._polys()
        self.orig_thetas = self.mesh.thetas.copy()
        self.orig_phis = self.mesh.phis.copy()
        self.orig_matrix_pipe.send(self.orig_matrix)
        self._v = self.orig_matrix.conj()[-1]
        self.curr_theta_idx = np.argmax(self.mesh.beta)
        self.curr_phi_idx = np.argmax(self.mesh.beta)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v: np.ndarray):
        if not self._v.ndim == 1:
            raise AttributeError('Require number of dimensions 1 (1-dimensional vector) to show propagation.')
        self._v = v
        self.update()

    def update(self):
        """Based on the current values in the mesh, update all plots accordingly.

        """
        v = self._v / np.linalg.norm(self._v)
        propagated_magnitudes = self.mesh.propagate(v)
        input_powers = np.abs(propagated_magnitudes[1])
        output_powers = np.abs(propagated_magnitudes[-1])
        powers = np.abs(propagated_magnitudes[1:, :].flatten()[self.path_idx])
        poly_powers = [powers[i] for i, multipoly in enumerate(self.path_array) for _ in multipoly]
        self.phi_pipe.send(self.mesh.phis)
        self.theta_pipe.send(self.mesh.thetas)
        self.power_pipe.send(poly_powers)
        self.input_pipe.send(input_powers)
        self.output_pipe.send(output_powers)
        curr_matrix = self.mesh.matrix()
        self.curr_matrix_pipe.send(self.mesh.matrix())
        self.fidelity_matrix_pipe.send(curr_matrix.conj() @ self.orig_matrix.T)

    def _polys(self):
        """Path array, which is useful for plotting and demo purposes.

        Returns:
            A numpy array in rail form consisting of shapely MultiPolygons (note: NOT numbers),
            which are waveguide segments that contain power information in the mesh,
            or :code:`None` where there are no waveguides along a rail.
        """
        path_array = []
        theta_array = np.full(self.mesh.num_nodes, None)
        phi_array = np.full(self.mesh.num_nodes, None)
        for column in self.mesh.columns:
            path_array.append([[None, None, None, None] for _ in range(self.n)])
            for node in column.nodes:
                mzi = self.mzis[(node.column, node.top, node.bottom)]
                top_port = mzi.port['a0'].copy
                top_port.a = 0
                bottom_port = mzi.port['a1'].copy
                bottom_port.a = 0
                top_path = mzi.path(flip=False).to(top_port)
                bottom_path = mzi.path(flip=True).to(bottom_port)
                top, bottom = top_path.layer_to_geoms[self.ridge], bottom_path.layer_to_geoms[self.ridge]
                phases = top_path.layer_to_geoms[self.ps.heater]
                num_geoms = len(top)
                num_geoms_bend = (num_geoms - 4) // 4
                path_array[-1][node.top] = [top[:num_geoms_bend + 1],
                                            top[num_geoms_bend + 1:num_geoms_bend + 2],
                                            top[num_geoms_bend + 2:num_geoms_bend * 3 + 3],
                                            top[num_geoms_bend * 3 + 3:]]
                path_array[-1][node.bottom] = [bottom[:num_geoms_bend + 1],
                                               bottom[num_geoms_bend + 1:num_geoms_bend + 2],
                                               bottom[num_geoms_bend + 2:num_geoms_bend * 3 + 3],
                                               bottom[num_geoms_bend * 3 + 3:]]
                theta_array[node.node_id] = phases[1]
                phi_array[node.node_id] = phases[0]

        # flatten to just a list of polygons
        x = np.hstack([np.asarray(pa, dtype=object) for pa in path_array]).T.flatten()

        # index to where there are waveguides that map to each polygon
        return x[x != None], theta_array, phi_array, np.where(x != None)

    def _io_plot(self, power_cmap: str):
        input_x = [self.port[f'a{i}'].x - self.interport_distance / 4 for i in range(self.n)]
        input_y = [self.port[f'a{i}'].y for i in range(self.n)]
        output_x = [self.port[f'b{i}'].x + self.interport_distance / 4 for i in range(self.n)]
        output_y = [self.port[f'b{i}'].y for i in range(self.n)]

        input_arrows = lambda data: hv.VectorField(
            (input_x, input_y, np.zeros_like(input_x), data)).opts(
            pivot='tip', line_width=4,
            magnitude=hv.dim('Magnitude') * self.interport_distance, rescale_lengths=False) * hv.VectorField(
            (input_x, input_y, np.zeros_like(input_x), data)).opts(
            pivot='tip', color='Magnitude', cmap=power_cmap, line_width=2,
            magnitude=hv.dim('Magnitude') * self.interport_distance, rescale_lengths=False, clim=(0, 1))
        output_arrows = lambda data: hv.VectorField(
            (output_x, output_y, np.zeros_like(output_x), data)).opts(
            pivot='tail', magnitude=hv.dim('Magnitude') * self.interport_distance, rescale_lengths=False,
            line_width=4) * hv.VectorField(
            (output_x, output_y, np.zeros_like(output_x), data)).opts(
            pivot='tail', color='Magnitude', magnitude=hv.dim('Magnitude') * self.interport_distance,
            rescale_lengths=False, cmap=power_cmap, line_width=2, clim=(0, 1))

        inputs = hv.DynamicMap(input_arrows, streams=[self.input_pipe])
        outputs = hv.DynamicMap(output_arrows, streams=[self.output_pipe])

        return inputs, outputs

    def add_random_error(self, error_std: float = 0.01):
        self.mesh = self.mesh.add_error_variance(error_std)
        self.update()

    def _power_plots(self, height: int, power_cmap: str, title: str):
        polys = [np.asarray(p.exterior.coords.xy).T for multipoly in self.path_array for p in multipoly]

        power_polys = lambda data: hv.Polygons(
            [{('x', 'y'): poly, 'amplitude': z} for poly, z in zip(polys, data)], vdims='amplitude'
        )

        status = lambda data: hv.Text(x=self.size[0] / 2, y=self.size[1] + self.interport_distance / 2, text=data)

        waveguides = hv.Polygons(polys).opts(data_aspect=1, frame_height=height,
                                             ylim=(-10, self.size[1] + 10),
                                             color='black', line_width=2)

        powers = hv.DynamicMap(power_polys, streams=[self.power_pipe]).opts(
            data_aspect=1, frame_height=height,
            ylim=(-self.interport_distance, self.size[1] + self.interport_distance),
            line_color='none', cmap=power_cmap,
            shared_axes=False, colorbar=True, clim=(0, 1),
            title=title, tools=['hover'],
            xlim=(-self.interport_distance * 2,
                  self.size[0] + self.interport_distance)
        )

        return waveguides * powers * hv.DynamicMap(status, streams=[self.status_pipe])

    def _phase_plots(self, height: int, theta_cmap: str, phi_cmap: str):
        theta_geoms = [np.asarray(p.buffer(-0.25).exterior.coords.xy).T for p in self.theta_array]
        phi_geoms = [np.asarray(p.buffer(-0.25).exterior.coords.xy).T for p in self.phi_array]
        theta_centroids = [(poly.centroid.x, poly.centroid.y) for poly in self.theta_array]
        phi_centroids = [(poly.centroid.x, poly.centroid.y) for poly in self.phi_array]

        theta_text = hv.Overlay([hv.Text(centroid[0], centroid[1] + self.interport_distance / 2,
                                         f'θ{i}', fontsize=7).opts(color='darkblue') for i, centroid in
                                 enumerate(theta_centroids)])
        phi_text = hv.Overlay([hv.Text(centroid[0], centroid[1] + self.interport_distance / 2,
                                       f'ϕ{i}', fontsize=7).opts(color='darkgreen') for i, centroid in
                               enumerate(phi_centroids)])
        theta_polys = lambda data: hv.Polygons(
            [{('x', 'y'): poly, 'theta': z} for poly, z in zip(theta_geoms, data)], vdims='theta'
        )
        theta_bg = hv.Polygons([np.asarray(p.exterior.coords.xy).T for p in self.theta_array]).opts(color='darkblue')
        phi_polys = lambda data: hv.Polygons(
            [{('x', 'y'): poly, 'phi': z} for poly, z in zip(phi_geoms, data)], vdims='phi'
        )
        phi_bg = hv.Polygons([np.asarray(p.exterior.coords.xy).T for p in self.phi_array]).opts(color='darkgreen')

        theta = hv.DynamicMap(theta_polys, streams=[self.theta_pipe]).opts(
            data_aspect=1, frame_height=height, line_color='none', cmap=theta_cmap,
            shared_axes=False, clim=(0, 2 * np.pi), tools=['hover', 'tap']
        )
        phi = hv.DynamicMap(phi_polys, streams=[self.phi_pipe]).opts(
            data_aspect=1, frame_height=height, line_color='none', cmap=phi_cmap,
            shared_axes=False, clim=(0, 2 * np.pi), tools=['hover', 'tap']
        )

        sel_theta = hv.streams.Selection1D(source=theta)
        sel_phi = hv.streams.Selection1D(source=phi)

        def change_theta(*events):
            for event in events:
                if event.name == 'value':
                    index = sel_theta.index[0] if len(sel_theta.index) > 0 else self.curr_theta_idx
                    self.mesh.thetas[index] = event.new * np.pi
                    self.theta_pipe.send(self.mesh.thetas)
                    self.update()

        def change_phi(*events):
            for event in events:
                if event.name == 'value':
                    index = sel_phi.index[0] if len(sel_phi.index) > 0 else self.curr_phi_idx
                    self.mesh.phis[index] = event.new * np.pi
                    self.phi_pipe.send(self.mesh.phis)
                    self.update()

        theta_set = pn.widgets.FloatSlider(start=0, end=2, step=0.01,
                                           value=0, name='θ / π', format='1[.]000')
        theta_set.param.watch(change_theta, 'value')
        theta_set.value = self.mesh.thetas[0] / np.pi
        phi_set = pn.widgets.FloatSlider(start=0, end=2, step=0.01,
                                         value=0, name='ϕ / π', format='1[.]000')
        phi_set.param.watch(change_phi, 'value')
        phi_set.value = self.mesh.phis[0] / np.pi

        def pi_formatter(value):
            return f'{value:.1f}π'

        def theta_plot_(index, data):
            index = index[0] if len(index) > 0 else self.curr_theta_idx
            x = np.linspace(0, 2 * np.pi, 1000)
            a, b = self.mesh.nodes[index].alpha, self.mesh.nodes[index].beta
            curve = hv.Curve((x / np.pi, beta_phase(x, a, b)))
            theta_set.value = data[index] / np.pi
            self.curr_theta_idx = index
            return hv.Spikes([data[index] / np.pi], label=f'θ{index}').opts(
                color='black', spike_length=beta_phase(data[index], a, b)) * curve.relabel(f'B({a},{b})')

        def phi_plot_(index, data):
            index = index[0] if len(index) > 0 else self.curr_phi_idx
            x = np.linspace(0, 2 * np.pi, 1000)
            curve = hv.Curve((x / np.pi, np.ones_like(x) / 2 * np.pi))
            phi_set.value = data[index] / np.pi
            self.curr_phi_idx = index
            return hv.Spikes([data[index] / np.pi], label=f'ϕ{index}').opts(
                color='black', spike_length=1 / 2 * np.pi) * curve.relabel(f'U(2π)')

        theta_plot = hv.DynamicMap(theta_plot_, streams=[sel_theta, self.theta_pipe]).opts(shared_axes=False,
                                                                                           xformatter=pi_formatter,
                                                                                           height=200)
        phi_plot = hv.DynamicMap(phi_plot_, streams=[sel_phi, self.phi_pipe]).opts(shared_axes=False,
                                                                                   xformatter=pi_formatter,
                                                                                   height=200)

        return theta_bg * theta_text * theta, phi_bg * phi_text * phi, theta_set, phi_set, theta_plot, phi_plot

    def _matrix_plots(self):

        def complex_matrix(data):
            return hv.VectorField((*np.meshgrid(np.arange(self.n), np.arange(self.n)),
                                   np.angle(data), np.abs(data))).opts(magnitude=hv.dim('Magnitude') * 0.5)

        status_ = lambda data: hv.Text(x=self.n / 2 - 0.5, y=self.n - 0.5, text=data)
        status = hv.DynamicMap(status_, streams=[self.status_pipe])

        orig_matrix = hv.DynamicMap(complex_matrix, streams=[self.orig_matrix_pipe], label='ideal').opts(title='Unitary comparison',
                                                                                          rescale_lengths=False,
                                                                                          xticks=np.arange(self.n),
                                                                                          yticks=np.arange(self.n),
                                                                                          invert_yaxis=True,
                                                                                          xlim=(-1, self.n),
                                                                                          ylim=(-1, self.n),
                                                                                          xlabel='Column',
                                                                                          ylabel='Row'
                                                                                          )
        curr_matrix = hv.DynamicMap(complex_matrix, streams=[self.curr_matrix_pipe], label='actual').opts(rescale_lengths=False,
                                                                                          xticks=np.arange(self.n),
                                                                                          yticks=np.arange(self.n),
                                                                                          color='red',
                                                                                          invert_yaxis=True,
                                                                                          xlim=(-1, self.n),
                                                                                          ylim=(-1, self.n),
                                                                                          xlabel='Column',
                                                                                          ylabel='Row'
                                                                                          )

        product = hv.DynamicMap(complex_matrix, streams=[self.fidelity_matrix_pipe]).opts(title='Fidelity unitary',
                                                                                          rescale_lengths=False,
                                                                                          xticks=np.arange(self.n),
                                                                                          yticks=np.arange(self.n),
                                                                                          invert_yaxis=True,
                                                                                          xlim=(-1, self.n),
                                                                                          ylim=(-1, self.n),
                                                                                          xlabel='Column',
                                                                                          ylabel='Row'
                                                                                          )
        return (curr_matrix * orig_matrix * status).opts(shared_axes=False), (product * status).opts(shared_axes=False)

    def hvsim(self, title='Mesh', height: int = 700, wide: bool = False,
              power_cmap: str = 'gray', theta_cmap='twilight', phi_cmap='twilight',
              self_configure_matrix: bool = True):
        powers = self._power_plots(height, power_cmap, title)
        inputs, outputs = self._io_plot(power_cmap)
        theta, phi, theta_set, phi_set, theta_plot, phi_plot = self._phase_plots(height, theta_cmap, phi_cmap)
        self.update()
        reset_button = pn.widgets.Button(name='Reset phases')
        reset_button.on_click(lambda *events: self.reset_phases())
        randomize_button = pn.widgets.Button(name='Randomize mesh')
        randomize_button.on_click(lambda *events: self.randomize())
        program_columns = pn.widgets.Button(name='Configure vector')
        program_columns.on_click(lambda *events: self.program_columns())
        if self_configure_matrix:
            self_configure_unitary = pn.widgets.Button(name='Configure unitary by column')
            self_configure_unitary.on_click(lambda *events: self.self_configure_unitary(by_column=True))
            self_configure_unitary_fast = pn.widgets.Button(name='Configure unitary by vector unit')
            self_configure_unitary_fast.on_click(lambda *events: self.self_configure_unitary(by_column=False))
            program_columns = pn.Column(program_columns, self_configure_unitary, self_configure_unitary_fast)
        plot = theta * phi * inputs * outputs * powers
        plot.opts(invert_yaxis=True)
        matrix = self._matrix_plots()
        phases = pn.Column(theta_set, theta_plot), pn.Column(phi_set, phi_plot)
        if wide:
            return pn.Column(plot,
                             pn.Row(pn.Tabs(('Phases', pn.Row(*phases)), ('Matrix', pn.Row(*matrix))),
                                    pn.Column(reset_button, randomize_button, program_columns)))
        else:
            return pn.Row(plot, pn.Column(pn.Tabs(('Phases', pn.Column(*phases)), ('Matrix', pn.Column(*matrix))),
                                          reset_button, randomize_button, program_columns))

    def reset_phases(self):
        self.mesh.thetas = self.orig_thetas.copy()
        self.mesh.phis = self.orig_phis.copy()
        self.update()

    def randomize(self):
        """Randomize the phases in this mesh and set the programmed vector to maximize the bottom output.

        Returns:

        """
        self.status_pipe.send('Haar-randomizing phases...')
        # self.mesh.thetas = self.mesh.rand_theta()
        # self.mesh.phis = np.random.rand(self.mesh.num_nodes) * 2 * np.pi
        self.orig_matrix = random_unitary(self.n)
        self.mesh = self.mesh_fn(self.orig_matrix)
        self.orig_thetas = self.mesh.thetas.copy()
        self.orig_phis = self.mesh.phis.copy()
        self.orig_matrix_pipe.send(self.orig_matrix)
        self.status_pipe.send('Propagating ideal vector v...')
        self.v = self.orig_matrix.conj()[-1]
        self.status_pipe.send('Idle')

    def program_columns(self):
        """For a self configurable architecture, configure the currently propagated vector.
        """
        self.status_pipe.send('Haar-randomizing phases...')
        self.mesh.thetas = self.mesh.rand_theta()
        self.mesh.phis = np.random.rand(self.mesh.num_nodes) * 2 * np.pi
        self.update()
        for column in self.mesh.columns:
            self.status_pipe.send(f'Tune ϕ in col {column.nodes[0].column} for v.')
            self.mesh.phis[(column.node_idxs,)] = self.orig_phis[(column.node_idxs,)]
            self.update()
            if self.n == 2:
                # hack for demo purposes
                time.sleep(1)
            self.status_pipe.send(f'Tune θ in col {column.nodes[0].column} for v.')
            self.mesh.thetas[(column.node_idxs,)] = self.orig_thetas[(column.node_idxs,)]
            self.update()
        self.status_pipe.send('Idle')

    def self_configure_unitary(self, by_column: bool):
        """For a self configurable architecture, configure each of the rows of the matrix stored in the desired mesh.
        """
        self.status_pipe.send(f'Haar-randomizing phases...')
        self.mesh.thetas = self.mesh.rand_theta()
        self.mesh.phis = np.random.rand(self.mesh.num_nodes) * 2 * np.pi
        self.update()
        offset = 0
        for i in reversed(range(self.n)):
            self.status_pipe.send(f'Sending input {i}.')
            self.v = self.orig_matrix[i].conj()
            node_idxs = self.mesh.node_idxs[offset:offset + i]
            offset += i
            mesh = ForwardMesh([self.mesh.nodes[idx] for idx in node_idxs])
            if by_column:
                for column in mesh.columns:
                    if len(column.node_idxs) > 0 or i == 0:
                        self.status_pipe.send(f'Tune ϕ in col {column.nodes[0].column} for u{i}.')
                        self.mesh.phis[(column.node_idxs,)] = self.orig_phis[(column.node_idxs,)]
                        self.update()
                        self.status_pipe.send(f'Tune θ in col {column.nodes[0].column} for u{i}.')
                        self.mesh.thetas[(column.node_idxs,)] = self.orig_thetas[(column.node_idxs,)]
                        self.update()
            else:
                self.status_pipe.send(f'Tune ϕ, θ for u{i}.')
                self.mesh.phis[(mesh.node_idxs,)] = self.orig_phis[(mesh.node_idxs,)]
                self.mesh.thetas[(mesh.node_idxs,)] = self.orig_thetas[(mesh.node_idxs,)]
                self.update()
        self.status_pipe.send('Idle')
