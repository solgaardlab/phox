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


def beta_pdf(x, a, b):
    return (x ** (b - 1) * (1 - x) ** (a - 1)) / beta_func(a, b)


def beta_phase(theta, a, b):
    x = np.cos(theta / 2) ** 2
    return np.abs(beta_pdf(x, a, b) * np.sin(theta / 2) * np.cos(theta / 2) / np.pi)


ps = ThermalPS(Waveguide((2, 20)), ps_w=10, via=Via((0.4, 0.4), 0.1))
dc = DC(waveguide_w=2, interaction_l=10, bend_l=5, interport_distance=25, gap_w=0.5)


@fix_dataclass_init_docs
@dataclass
class Mesh(Device):
    """Default rectangular mesh, or triangular mesh if specified
    Note: triangular meshes can self-configure, but rectangular meshes cannot.

    Attributes:
        mzi: The :code:`MZI` object, which acts as the unit cell for the mesh.
        n: The number of inputs and outputs for the mesh.
        triangular: Triangular mesh, otherwise rectangular mesh
        name: Name of the device
    """

    mesh: ForwardMesh
    dc: DC = dc
    ps: ThermalPS = ps
    ridge: str = CommonLayer.RIDGE_SI
    name: str = 'mesh'

    def __post_init_post_parse__(self):
        self.interport_distance = self.dc.interport_distance
        port = {}
        self.n = self.mesh.n
        pattern_to_layer = []
        self.mzis = {}
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
        self.path_array, self.theta_array, self.phi_array, self.path_idx = self._polys()
        self._v = np.ones(self.n, dtype=np.complex128) / np.sqrt(self.n)
        self.orig_thetas = self.mesh.thetas.copy()
        self.orig_phis = self.mesh.phis.copy()

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

    def hvsim(self, title='Mesh', height: int = 700, wide: bool = False,
              power_cmap: str = 'hot', theta_cmap='twilight', phi_cmap='twilight'):
        polys = [np.asarray(p.exterior.coords.xy).T for multipoly in self.path_array for p in multipoly]

        waveguides = hv.Polygons(polys).opts(data_aspect=1, frame_height=height,
                                             ylim=(-10, self.size[1] + 10),
                                             color='black', line_width=2)
        theta_geoms = [np.asarray(p.buffer(1).exterior.coords.xy).T for p in self.theta_array]
        phi_geoms = [np.asarray(p.buffer(1).exterior.coords.xy).T for p in self.phi_array]
        theta_centroids = [(poly.centroid.x, poly.centroid.y) for poly in self.theta_array]
        phi_centroids = [(poly.centroid.x, poly.centroid.y) for poly in self.phi_array]

        theta_text = hv.Overlay([hv.Text(centroid[0], centroid[1] + self.interport_distance / 2,
                                         f'θ{i}', fontsize=7).opts(color='darkblue') for i, centroid in
                                 enumerate(theta_centroids)])
        phi_text = hv.Overlay([hv.Text(centroid[0], centroid[1] + self.interport_distance / 2,
                                       f'ϕ{i}', fontsize=7).opts(color='darkgreen') for i, centroid in
                               enumerate(phi_centroids)])

        power_polys = lambda data: hv.Polygons(
            [{('x', 'y'): poly, 'amplitude': z} for poly, z in zip(polys, data)], vdims='amplitude'
        )
        theta_polys = lambda data: hv.Polygons(
            [{('x', 'y'): poly, 'theta': z} for poly, z in zip(theta_geoms, data)], vdims='theta'
        )
        phi_polys = lambda data: hv.Polygons(
            [{('x', 'y'): poly, 'phi': z} for poly, z in zip(phi_geoms, data)], vdims='phi'
        )

        powers = hv.DynamicMap(power_polys, streams=[self.power_pipe]).opts(
            data_aspect=1, frame_height=height, ylim=(-10, self.size[1] + 10), line_color='none', cmap=power_cmap,
            shared_axes=False
        )
        theta = hv.DynamicMap(theta_polys, streams=[self.theta_pipe]).opts(
            data_aspect=1, frame_height=height, ylim=(-10, self.size[1] + 10), line_color='none', cmap=theta_cmap,
            shared_axes=False, clim=(0, 2 * np.pi), tools=['hover']
        )
        phi = hv.DynamicMap(phi_polys, streams=[self.phi_pipe]).opts(
            data_aspect=1, frame_height=height, ylim=(-10, self.size[1] + 10), line_color='none', cmap=phi_cmap,
            shared_axes=False, clim=(0, 2 * np.pi), tools=['hover']
        )

        sel_theta = hv.streams.Selection1D(source=theta)
        sel_phi = hv.streams.Selection1D(source=phi)

        def change_theta(*events):
            for event in events:
                if event.name == 'value':
                    self.mesh.thetas[sel_theta.index] = event.new * np.pi
                    self.theta_pipe.send(self.mesh.thetas)
                    self._propagate()

        def change_phi(*events):
            for event in events:
                if event.name == 'value':
                    self.mesh.phis[sel_phi.index] = event.new * np.pi
                    self.phi_pipe.send(self.mesh.phis)
                    self._propagate()

        theta_set = pn.widgets.FloatSlider(start=0, end=2, step=0.01,
                                           value=0, name='θ / π', format='1[.]000')
        theta_set.param.watch(change_theta, 'value')
        phi_set = pn.widgets.FloatSlider(start=0, end=2, step=0.01,
                                         value=0, name='ϕ / π', format='1[.]000')
        phi_set.param.watch(change_phi, 'value')

        self._propagate()
        self.theta_pipe.send(self.mesh.thetas)
        self.phi_pipe.send(self.mesh.phis)

        reset_button = pn.widgets.Button(name='Reset phases')
        reset_button.on_click(lambda *events: self.reset_phases())

        plot = theta * theta_text * phi * phi_text * waveguides * powers.options(colorbar=True, clim=(0, 1),
                                                                                 title=title,
                                                                                 tools=['hover', 'tap'])

        def pi_formatter(value):
            return f'{value:.1f}π'

        def theta_plot_(index, data):
            index = index[0] if len(index) > 0 else np.argmax(self.mesh.beta)
            x = np.linspace(0, 2 * np.pi, 1000)
            a, b = self.mesh.nodes[index].alpha, self.mesh.nodes[index].beta
            curve = hv.Curve((x / np.pi, beta_phase(x, a, b)))
            theta_set.value = data[index] / np.pi
            return hv.Spikes([data[index] / np.pi], label=f'θ{index}').opts(
                color='black', spike_length=beta_phase(data[index], a, b)) * curve.relabel(f'B({a},{b})')

        def phi_plot_(index, data):
            index = index[0] if len(index) > 0 else 0
            x = np.linspace(0, 2 * np.pi, 1000)
            curve = hv.Curve((x / np.pi, np.ones_like(x) / 2 * np.pi))
            phi_set.value = data[index] / np.pi
            return hv.Spikes([data[index] / np.pi], label=f'ϕ{index}').opts(
                color='black', spike_length=1 / 2 * np.pi) * curve.relabel(f'U(2π)')

        theta_plot = hv.DynamicMap(theta_plot_, streams=[sel_theta, self.theta_pipe]).opts(shared_axes=False,
                                                                                           xformatter=pi_formatter)
        phi_plot = hv.DynamicMap(phi_plot_, streams=[sel_phi, self.phi_pipe]).opts(shared_axes=False,
                                                                                   xformatter=pi_formatter)

        if wide:
            return pn.Column(plot, pn.Row(pn.Column(theta_set, theta_plot), pn.Column(phi_set, phi_plot), reset_button))
        else:
            return pn.Row(plot, pn.Column(theta_plot, theta_set, phi_plot, phi_set, reset_button))

    def reset_phases(self):
        self.mesh.thetas = self.orig_thetas.copy()
        self.mesh.phis = self.orig_phis.copy()
        self.theta_pipe.send(self.mesh.thetas)
        self.phi_pipe.send(self.mesh.phis)
        self._propagate()

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v: np.ndarray):
        if not self._v.ndim == 1:
            raise AttributeError('Require number of dimensions 1 (1-dimensional vector) to show propagation.')
        self._v = v
        self._propagate()

    def _propagate(self):
        v = self._v / np.linalg.norm(self._v)
        powers = np.abs(self.mesh.propagate(v)[1:, :].flatten()[self.path_idx])
        poly_powers = [powers[i] for i, multipoly in enumerate(self.path_array) for _ in multipoly]
        self.power_pipe.send(poly_powers)
