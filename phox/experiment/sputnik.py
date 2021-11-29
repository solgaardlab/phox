from typing import Tuple, Callable, Optional, Dict

from holoviews import opts
from holoviews.streams import Pipe
from tornado import gen
from tornado.ioloop import PeriodicCallback

from .activephotonicsimager import ActivePhotonicsImager, _get_grating_spot
from ..instrumentation import XCamera
from dphox.active import LocalMesh
from dphox.demo import mzi

import time
import numpy as np

from simphox.circuit import triangular, unbalanced_tree
from scipy.stats import unitary_group
from simphox.utils import random_unitary, random_vector
from ..utils import phases_to_vector
import panel as pn
import pickle

import logging
import holoviews as hv

from ..model.phase import PhaseCalibration

logger = logging.getLogger()
logger.setLevel(logging.WARN)

mesh = LocalMesh(mzi, 6)
PS_LAYER = 'heater'

SPUTNIK_CONFIG = {
    "network": {"theta_left": [[1, 1], [3, 2], [5, 3], [7, 4]], "phi_left": [[2, 1], [4, 2], [6, 3], [8, 4]],
                "theta_right": [[17, 1], [15, 2], [13, 3], [11, 4]], "phi_right": [[16, 1], [14, 2], [12, 2], [10, 4]],
                "theta_mesh": [[5, 0], [7, 1], [9, 2], [9, 0], [11, 1], [13, 0]],
                "phi_mesh": [[4, 0], [6, 1], [8, 2], [8, 0], [10, 1], [12, 0]], "theta_ref": [9, 5],
                "theta_rows": [[[1, 1], [5, 0], [9, 0], [13, 0], [17, 1]], [[3, 2], [7, 1], [11, 1], [15, 2]],
                               [[5, 3], [9, 2], [13, 3]], [[7, 4], [11, 4]], [[9, 5]]]},
    "thetas": [{"grid_loc": [1, 1], "spot_loc": [0, 0], "voltage_channel": 5, "meta_ps": []},
               {"grid_loc": [5, 0], "spot_loc": [4, 0], "voltage_channel": 17, "meta_ps": []},
               {"grid_loc": [9, 0], "spot_loc": [8, 0], "voltage_channel": 39, "meta_ps": []},
               {"grid_loc": [13, 0], "spot_loc": [12, 0], "voltage_channel": 48, "meta_ps": []},
               {"grid_loc": [17, 1], "spot_loc": [16, 0], "voltage_channel": 59, "meta_ps": []},
               {"grid_loc": [3, 2], "spot_loc": [2, 1], "voltage_channel": 7, "meta_ps": []},
               {"grid_loc": [7, 1], "spot_loc": [6, 1], "voltage_channel": 26, "meta_ps": []},
               {"grid_loc": [11, 1], "spot_loc": [10, 1], "voltage_channel": 40, "meta_ps": []},
               {"grid_loc": [15, 2], "spot_loc": [14, 1], "voltage_channel": 55, "meta_ps": []},
               {"grid_loc": [5, 3], "spot_loc": [4, 2], "voltage_channel": 15, "meta_ps": []},
               {"grid_loc": [9, 2], "spot_loc": [8, 2], "voltage_channel": 32, "meta_ps": []},
               {"grid_loc": [13, 3], "spot_loc": [12, 2], "voltage_channel": 46, "meta_ps": []},
               {"grid_loc": [7, 4], "spot_loc": [6, 3], "voltage_channel": 25, "meta_ps": []},
               {"grid_loc": [11, 4], "spot_loc": [10, 3], "voltage_channel": 38, "meta_ps": []},
               {"grid_loc": [9, 5], "spot_loc": [8, 4], "voltage_channel": 28, "meta_ps": []}],
    "phis": [{"grid_loc": [2, 1], "spot_loc": [4, 0], "voltage_channel": 4, "meta_ps": [[1, 1], [5, 0]]},
             {"grid_loc": [4, 0], "spot_loc": [4, 0], "voltage_channel": 13, "meta_ps": [[1, 1], [5, 0]]},
             {"grid_loc": [6, 1], "spot_loc": [8, 0], "voltage_channel": 20, "meta_ps": [[5, 0], [9, 0]]},
             {"grid_loc": [8, 0], "spot_loc": [8, 0], "voltage_channel": 27, "meta_ps": [[5, 0], [9, 0]]},
             {"grid_loc": [10, 1], "spot_loc": [12, 0], "voltage_channel": 37, "meta_ps": [[9, 0], [13, 0]]},
             {"grid_loc": [12, 0], "spot_loc": [12, 0], "voltage_channel": 44, "meta_ps": [[9, 0], [13, 0]]},
             {"grid_loc": [16, 1], "spot_loc": [16, 0], "voltage_channel": 56, "meta_ps": [[13, 0], [17, 1]]},
             {"grid_loc": [4, 2], "spot_loc": [6, 1], "voltage_channel": 16, "meta_ps": [[3, 2], [7, 1]]},
             {"grid_loc": [8, 2], "spot_loc": [10, 1], "voltage_channel": 35, "meta_ps": [[7, 1], [11, 1]]},
             {"grid_loc": [12, 2], "spot_loc": [14, 1], "voltage_channel": 47, "meta_ps": [[11, 1], [15, 2]]},
             {"grid_loc": [14, 2], "spot_loc": [14, 1], "voltage_channel": 51, "meta_ps": [[11, 1], [15, 2]]},
             {"grid_loc": [6, 3], "spot_loc": [8, 2], "voltage_channel": 21, "meta_ps": [[5, 3], [9, 2]]},
             {"grid_loc": [8, 4], "spot_loc": [10, 3], "voltage_channel": 24, "meta_ps": [[7, 4], [11, 4]]},
             {"grid_loc": [10, 4], "spot_loc": [10, 3], "voltage_channel": 34, "meta_ps": [[7, 4], [11, 4]]}]}


class Sputnik(ActivePhotonicsImager):
    def __init__(self, interlayer_xy: Tuple[float, float], spot_xy: Tuple[int, int], interspot_xy: Tuple[int, int],
                 ps_calibration: Dict, window_shape: Tuple[int, int] = (15, 10),
                 backward_shift: float = 0.033, home: Tuple[float, float] = (0, 0), stage_port: str = '/dev/ttyUSB1',
                 laser_port: str = '/dev/ttyUSB0', lmm_port: str = None,
                 camera_calibration_filepath: Optional[str] = None, integration_time: int = 20000,
                 plim: Tuple[float, float] = (0.05, 4.25), vmax: float = 6):
        """This class is meant to test our first triangular mesh fabricated in AMF.
        These chips are 6x6, and they contain

        Args:
            interlayer_xy:
            spot_xy:
            interspot_xy:
            window_shape:
            backward_shift:
            home:
            stage_port:
            laser_port:
            lmm_port:
            camera_calibration_filepath:
            integration_time:
            plim:
            vmax:
        """
        self.network = SPUTNIK_CONFIG['network']
        self.thetas = [PhaseShifter(**ps_dict, mesh=self,
                                    calibration=PhaseCalibration(**ps_calibration[tuple(ps_dict['grid_loc'])])
                                    ) for ps_dict in SPUTNIK_CONFIG['thetas']]
        self.thetas: Dict[Tuple[int, int], PhaseShifter] = {ps.grid_loc: ps for ps in self.thetas}
        self.phis = [PhaseShifter(**ps_dict, mesh=self,
                                  calibration=PhaseCalibration(**ps_calibration[tuple(ps_dict['grid_loc'])])
                                  ) for ps_dict in SPUTNIK_CONFIG['phis']]
        self.phis: Dict[Tuple[int, int], PhaseShifter] = {ps.grid_loc: ps for ps in self.phis}
        self.ps: Dict[Tuple[int, int], PhaseShifter] = {**self.thetas, **self.phis}
        self.interlayer_xy = interlayer_xy
        self.spot_xy = s = spot_xy
        self.interspot_xy = ixy = interspot_xy

        self.spots = [(j * ixy[0] + s[0], i * ixy[1] + s[1], window_shape[0], window_shape[1])
                      for j in range(6) for i in range(3)]
        self.camera = XCamera(integration_time=integration_time, spots=self.spots)
        self.integration_time = integration_time
        self.backward = False
        self.backward_shift = backward_shift
        super(Sputnik, self).__init__(home, stage_port, laser_port, lmm_port, camera_calibration_filepath,
                                      integration_time, plim, vmax)

        self.reset_control()
        self.camera.start_frame_loop()
        self.go_home()
        self.stage.wait_until_stopped()
        self.power_pipe = Pipe()
        self.ps_pipe = Pipe()
        self.spot_pipe = Pipe(data=[(i, 0) for i in range(6)])
        time.sleep(0.1)

    def to_layer(self, layer: int):
        self.stage.move(x=self.home[0] + self.interlayer_xy[0] * layer,
                        y=self.home[1] + self.interlayer_xy[1] * layer + self.backward * self.backward_shift)
        self.stage.wait_until_stopped()

    def mesh_img(self, n: int, wait_time: float = 0.5, window_size: int = 20):
        """

        Args:
            n: Number of inputs to the mesh
            wait_time: Wait time after the stage stops moving for things to settle
            window_size: Window size for the spots

        Returns:

        """
        powers = []
        spots = []
        s, ixy = self.spot_xy, self.interspot_xy
        for m in range(n + 1):
            self.to_layer(3 * m if m < n else 3 * n - 2)
            time.sleep(wait_time)
            img = self.camera.frame()
            if m < n:
                powers.append(
                    np.hstack([np.vstack([_get_grating_spot(img, center=(j * ixy[0] + s[0], i * ixy[1] + s[1]),
                                                            window_size=window_size)[0]
                                          for j in range(n)]) for i in range(3)]))
                spots.append(np.hstack([np.vstack([_get_grating_spot(img, center=(j * ixy[0] + s[0], i * ixy[1] + s[1]),
                                                                     window_size=window_size)[1] / np.sum(
                    powers[-1][:, i])
                                                   for j in range(n)]) for i in range(3)]))
            else:
                powers.append(np.vstack([_get_grating_spot(img, center=(j * ixy[0] + s[0], s[1]),
                                                           window_size=window_size)[0] for j in range(n)]))
                spots.append(np.vstack([_get_grating_spot(img, center=(j * ixy[0] + s[0], s[1]),
                                                          window_size=window_size)[1] / np.sum(powers[-1])
                                        for j in range(n)]))
        return np.fliplr(np.hstack(powers[::-1])), np.fliplr(np.hstack(spots[::-1]))

    def sweep(self, channel: int, layer: int, vlim: Tuple[float, float],
              wait_time: float = 0.0, n_samples: int = 1001, move: bool = True,
              pbar: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            channel: voltage channel to sweep
            layer: layer to move the stage
            vlim: Voltage limit for the sweep
            wait_time: Wait time between setting the temperature and taking the image
            n_samples: Number of samples
            move: whether to move to the appropriate layer
            pbar: progress bar (optional) to track the progress of the sweep

        Returns:

        """
        if move:
            self.to_layer(layer)
        vs = np.sqrt(np.linspace(vlim[0] ** 2, vlim[1] ** 2, n_samples))
        iterator = pbar(vs) if pbar is not None else vs
        powers = []
        for v in iterator:
            self.control.write_chan(channel, v)
            time.sleep(wait_time)
            powers.append(self.camera.spot_powers)
        return vs, np.asarray(powers).T

    def reset_control(self, vmin: float = 2):
        for device in self.control.system.devices:
            device.reset_device()
        for ps in self.ps:
            self.control.write_chan(self.ps[ps].voltage_channel, vmin)

    def propagation_toggle_panel(self, chan: int = 64):
        def toggle(*events):
            self.toggle_propagation_direction(chan)

        button = pn.widgets.Button(name='Switch Propagation Direction')
        button.on_click(toggle)
        return button

    def toggle_propagation_direction(self, chan: int = 64):
        self.backward = not self.backward
        self.control.ttl_toggle(chan)

    def led_panel(self, chan: int = 65):
        return self.control.continuous_slider(chan, name='LED Voltage', vlim=(0, 1))

    def home_panel(self):
        home_button = pn.widgets.Button(name='Home')

        def go_home(*events):
            self.go_home()

        home_button.on_click(go_home)
        return home_button

    def input_panel(self):
        def transparent_bar(*events):
            self.set_transparent()

        def transparent_cross(*events):
            self.set_transparent(bar=False)

        def reset(*events):
            self.reset_control()

        def alternating(*events):
            self.set_input(np.asarray((1, 0, 1, 0, 1)))

        def uniform(*events):
            self.set_input(np.asarray((1, 1, 1, 1, 1)))

        def random(*events):
            self.set_rand_unitary()

        def basis(i: int):
            def f(*events):
                self.set_input(np.asarray(np.eye(5)[i]))

            return f

        bar_button = pn.widgets.Button(name='Transparent (Bar)')
        bar_button.on_click(transparent_bar)
        cross_button = pn.widgets.Button(name='Transparent (Cross)')
        cross_button.on_click(transparent_cross)
        alternating_button = pn.widgets.Button(name='Alternate In (1-0-1-0-1)')
        alternating_button.on_click(alternating)
        uniform_button = pn.widgets.Button(name='Uniform In (1-1-1-1-1)')
        uniform_button.on_click(uniform)
        random_button = pn.widgets.Button(name='Random Unitary')
        random_button.on_click(random)
        button_list = [pn.widgets.Button(name=f'{i}', width=15) for i in range(5)]
        for i, button in enumerate(button_list):
            button.on_click(basis(i))
        buttons = pn.Row(*button_list)
        reset_button = pn.widgets.Button(name='Zero Voltages')
        reset_button.on_click(reset)
        return pn.Column(reset_button, bar_button, cross_button, alternating_button, uniform_button, buttons)

    def set_unitary(self, u: np.ndarray):
        network = triangular(u)
        thetas, phis, gammas = network.params
        # thetas, phis, _, gammas = reck(np.fliplr(np.flipud(u)))
        self.set_unitary_phases(np.mod(thetas, 2 * np.pi), np.mod(phis, 2 * np.pi))
        return gammas
        # network = triangular(u.conj().T)
        # thetas, phis, gammas = network.params
        # self.set_unitary_phases(np.mod(thetas, 2 * np.pi), np.mod(phis, 2 * np.pi))
        # return gammas

    def uhash(self, x: np.ndarray, us: np.ndarray, uc: Optional[np.ndarray] = None):
        targets = []
        preds = []
        uc = np.eye(4) if uc is None else uc
        for i in range(16):
            target = []
            pred = []
            for j in range(16):
                v = uc @ np.asarray(x[4 * j:4 * (j + 1)])
                u = us[i, j] @ uc.conj().T
                self.set_unitary(u.conj().T)
                self.set_input(np.hstack((v, 0)))
                time.sleep(0.1)
                target.append(np.abs(self.fractional_right[:4]))
                pred.append(np.abs(us[i, j] @ v) ** 2)
            targets.append(np.hstack(target))
            preds.append(np.hstack(pred))

    def set_rand_unitary(self):
        alphas = np.asarray([1, 2, 1, 3, 2, 1])
        thetas = 2 * np.arccos(np.power(np.random.rand(len(alphas)), 1 / (2 * alphas)))
        phis = np.random.rand(len(alphas)) * 2 * np.pi
        for ps_loc, phase in zip(self.network['theta_mesh'], thetas):
            self.ps[tuple(ps_loc)].phase = phase
        for ps_loc, phase in zip(self.network['phi_mesh'], phis):
            self.ps[tuple(ps_loc)].phase = phase

    def to_output(self):
        self.to_layer(0 if self.backward else 16)

    def u_fidelity(self, u: np.ndarray):
        vals = []
        for i in range(4):
            self.set_input((np.hstack((u.T[i], 0))))
            time.sleep(0.1)
            vals.append(self.fractional_right[i])
        return np.asarray(vals)

    def matvec_comparison(self, u: np.ndarray, v: np.ndarray, wait_time: float = 0.1):
        v = v / np.linalg.norm(v)
        self.set_input(np.hstack((v, 0)))
        self.set_unitary(u.conj().T)
        time.sleep(wait_time)
        res = self.fractional_right[:4]
        actual = np.abs(u @ v) ** 2
        return res, actual

    def haar_fidelities(self, n: int = 1000, pbar: Optional[Callable] = None):
        self.set_transparent()
        self.to_layer(16)
        iterator = pbar(range(n)) if pbar is not None else range(n)
        return np.asarray([self.u_fidelity(random_unitary(4)) for _ in iterator])

    def matvec_comparisons(self, n: int = 100, wait_time: float = 0.1, pbar: Optional[Callable] = None):
        self.set_transparent()
        self.to_layer(16)
        iterator = pbar(range(n)) if pbar is not None else range(n)
        return np.asarray([self.matvec_comparison(random_unitary(4),
                                                  random_vector(4), wait_time) for _ in iterator])

    def set_unitary_sc(self, u: np.ndarray, show_mesh: bool = False):
        t, p = self.network['theta_mesh'], self.network['phi_mesh']
        theta_mesh_list, phi_mesh_list = [[t[0], t[1], t[2]], [t[3], t[4]], t[5:]], \
                                         [[p[0], p[1], p[2]], [p[3], p[4]], p[5:]]

        self.set_unitary(u)
        # only need to shine the first three vectors
        for i, vector in enumerate(u.T[:3]):
            self.set_input(np.hstack((vector, 0)))
            for theta, phi in zip(theta_mesh_list[i], phi_mesh_list[i]):
                theta, phi = tuple(theta), tuple(phi)
                self.ps[phi].opt_spot(spot=(phi[0], phi[1] + 1), wait_time=0.01, maximize=True)
                self.ps[theta].opt_spot(spot=(phi[0], phi[1] + 1), wait_time=0.01, maximize=True)
                if show_mesh:
                    self.update_mesh_image()

    def mesh_panel(self, power_cmap: str = 'hot', ps_cmap: str = 'greens'):
        polys = [np.asarray(p.exterior.coords.xy).T
                 for multipoly in mesh.path_array.flatten()
                 for p in multipoly]
        waveguides = hv.Polygons(polys).opts(data_aspect=1, frame_height=200,
                                             ylim=(-10, 70), xlim=(0, mesh.size[0]),
                                             color='black', line_width=2)
        phase_shift_polys = [np.asarray(p.buffer(1).exterior.coords.xy).T
                             for p in mesh.phase_shifter_array('heater')]
        labels = np.fliplr(np.fliplr(np.mgrid[0:6, 0:19]).reshape((2, -1)).T)
        centroids = [(poly.centroid.x, poly.centroid.y) for poly in mesh.phase_shifter_array('heater')]

        text = hv.Overlay([hv.Text(centroid[0],
                                   centroid[1] + mesh.interport_distance / 2,
                                   f'{label[0]},{label[1]}', fontsize=7)
                           for label, centroid in zip(list(labels), centroids) if tuple(label) in self.ps])

        power_polys = lambda data: hv.Polygons(
            [{('x', 'y'): poly, 'power': z} for poly, z in zip(polys, data)], vdims='power'
        )
        ps_polys = lambda data: hv.Polygons(
            [{('x', 'y'): poly, 'phase_shift': z} for poly, z in zip(phase_shift_polys, data)], vdims='phase_shift'
        )
        powers = hv.DynamicMap(power_polys, streams=[self.power_pipe]).opts(
            data_aspect=1, frame_height=200, ylim=(-10, 70),
            xlim=(0, mesh.size[0]), line_color='none', cmap=power_cmap, shared_axes=False
        )
        ps = hv.DynamicMap(ps_polys, streams=[self.ps_pipe]).opts(
            data_aspect=1, frame_height=200, ylim=(-10, 70),
            xlim=(0, mesh.size[0]), line_color='none', cmap=ps_cmap, shared_axes=False, clim=(0, 2 * np.pi)
        )
        self.power_pipe.send(np.full(len(polys), np.nan))
        self.ps_pipe.send(np.full(len(phase_shift_polys), np.nan))

        def mesh_image(*events):
            self.update_mesh_image()

        image_button = pn.widgets.Button(name='Mesh Image')
        image_button.on_click(mesh_image)

        def read_output(*events):
            self.read_output(update_mesh=True)

        read_button = pn.widgets.Button(name='Self-Configure (Read) Output')
        read_button.on_click(read_output)

        return pn.Column(ps * waveguides * powers * text, pn.Row(image_button, read_button))

    def update_mesh_image(self):
        powers, spots = self.mesh_img(6)
        powers[powers <= 0] = 0
        powers = np.flipud(np.sqrt(powers / np.max(powers)))
        self.power_pipe.send([p for p, multipoly in zip(powers.flatten(), mesh.path_array.flatten())
                              for _ in multipoly])
        data = np.zeros((6, 19))
        for loc in self.ps:
            data[loc[1], loc[0]] = self.ps[loc].phase
        self.ps_pipe.send(np.flipud(data).flatten())

    def set_transparent(self, bar: bool = True, theta_only: bool = False):
        """

        Args:
            bar:
            theta_only:

        Returns:

        """
        for ps in self.thetas if theta_only else {**self.thetas, **self.phis}:
            self.ps[ps].phase = np.pi if bar else 0

    def calibrate_thetas(self, pbar: Optional[Callable] = None, n_samples=20000, wait_time: float = 0):
        """Row-wise calibration of the :math:`\\theta` phase shifters

        Args:
            pbar: Progress bar to keep track of each calibration

        Returns:
            Voltages used for the calibration and the resulting powers

        """
        self.reset_control()
        idx = 0
        iterator = self.thetas.values() if pbar is None else pbar(self.thetas.values())
        for ps in iterator:
            print(ps.grid_loc)
            if ps.spot_loc[1] > idx:
                input_ps = tuple(self.network['theta_left'][idx])
                self.ps[input_ps].phase = 0
                idx += 1
            ps.calibrate(pbar, n_samples=n_samples, wait_time=wait_time)
            ps.phase = np.pi

    def calibrate_phis(self, pbar: Optional[Callable] = None, n_samples=20000, wait_time: float = 0):
        self.reset_control()
        # since the thetas are calibrated
        self.set_transparent(theta_only=True)
        idx = 0
        iterator = self.phis.values() if pbar is None else pbar(self.phis.values())
        for ps in iterator:
            print(ps.grid_loc)
            if ps.spot_loc[1] > idx:
                input_ps = tuple(self.network['theta_left'][idx])
                self.ps[input_ps].phase = 0
                idx += 1
            ps.calibrate(pbar, n_samples=n_samples, wait_time=wait_time)
            ps.phase = 0

    def set_input(self, vector: np.ndarray, add_normalization: bool = False, theta_only: bool = False,
                  backward: bool = False):
        n = 4
        if vector.size == 4:
            vector = np.hstack((vector, 0))
        if add_normalization:
            vector = vector / np.sqrt(np.sum(np.abs(vector))) * np.sqrt(n / (n + 1))
            vector = np.append(vector, np.sqrt(1 / (n + 1)))

        mesh = unbalanced_tree(vector[::-1].astype(np.complex128))
        thetas, phis, gammas = mesh.params
        thetas, phis = np.mod(thetas[::-1], 2 * np.pi), np.mod(phis[::-1], 2 * np.pi)

        if self.backward or backward:
            # hack that fixes an unfortunate circuit design error.
            phis[1] = np.mod(phis[1] + phis[2], 2 * np.pi)
            phis[2] = np.mod(-phis[2], 2 * np.pi)

        for i in range(n):
            phase = {'theta': thetas[i], 'phi': phis[i]}
            for var in ('theta',) if theta_only else ('theta', 'phi'):
                key = f'{var}_right' if self.backward or backward else f'{var}_left'
                self.set_phase(self.network[key][i], phase[var])
        self.set_phase(self.network['theta_ref'], np.pi)
        return gammas[-1]

    def set_output(self, vector: np.ndarray):
        return self.set_input(vector, backward=True)

    def set_phase(self, ps, phase):
        self.ps[tuple(ps)].phase = phase

    @property
    def phases(self):
        return {loc: self.ps[loc].phase for loc in self.ps}

    def read_output(self, pbar: Optional[Callable] = None, update_mesh: bool = False):
        direction = 'left' if self.backward else 'right'
        theta_ps = self.network[f'theta_{direction}'][::-1]
        phi_ps = self.network[f'phi_{direction}'][::-1]

        theta_vs, phi_vs = [], []

        lower_phi = (0, 0, 1, 0, 0) if self.backward else (0, 0, 0, 0, 0)
        lower_theta = (0, 0, 0, 0, 0)

        self.to_layer(0 if self.backward else 16)

        # TODO(sunil): fix hardcoding
        for i in range(4):
            theta, phi = tuple(theta_ps[i]), tuple(phi_ps[i])
            # self.ps[theta].phase = np.pi / 2
            # time.sleep(0.1)
            self.to_layer(phi[0])
            time.sleep(0.1)
            idx = phi[1]
            self.ps[theta].phase = 2 * np.arctan2(
                np.sqrt(self.fractional_left[idx]), np.sqrt(self.fractional_left[idx + 1])
            )
            time.sleep(0.1)
            phi_p = 2 * np.arctan2(
                np.sqrt(self.fractional_right[idx]), np.sqrt(self.fractional_right[idx + 1])
            )
            possible_p = (phi_p, 2 * np.pi - phi_p)
            powers = np.zeros(2)
            for i, p in enumerate(possible_p):
                self.ps[phi].phase = p
                time.sleep(0.1)
                powers[i] = self.fractional_right[idx + 1]
            null_idx = 0 if powers[0] < powers[1] else 1
            self.ps[phi].phase = possible_p[null_idx]

            if update_mesh:
                self.update_mesh_image()

        thetas = np.asarray(theta_vs)  # change this based on calibration
        phis = np.asarray(phi_vs)  # change this based on calibration

        return phases_to_vector(thetas, phis, lower_theta, lower_phi)

    def calibrate_panel(self, vlim: Tuple[float, float] = (0.5, 4.5)):
        vs = np.sqrt(np.linspace(vlim[0] ** 2, vlim[1] ** 2, 20000))
        ps_dropdown = pn.widgets.Select(
            name="Phase Shifter", options=[f"{ps[0]}, {ps[1]}" for ps in self.ps], value="1, 1"
        )
        calibrated_values = pn.Row(
            hv.Overlay([hv.Curve((ps.v2p(vs) / np.pi * 2, vs)) for _, ps in self.ps.items()]).opts(
                xlabel='Phase (θ)', ylabel='Voltage (V)').opts(shared_axes=False, title='Calibration Curves',
                                                               xformatter='%fπ'),
            hv.Overlay([hv.Curve((vs, ps.v2p(vs) / np.pi * 2)) for _, ps in self.ps.items()]).opts(
                ylabel='Phase (θ)', xlabel='Voltage (V)').opts(shared_axes=False, yformatter='%fπ')
        )

        def to_layer(*events):
            ps_tuple = tuple([int(c) for c in ps_dropdown.value.split(', ')])
            self.to_layer(self.ps[ps_tuple].grid_loc[0])

        @pn.depends(ps_dropdown.param.value)
        def calibration_image(value):
            ps_tuple = tuple([int(c) for c in value.split(', ')])
            p = self.ps[ps_tuple].calibration
            vs_cal = np.sqrt(np.linspace(vlim[0] ** 2, vlim[1] ** 2, len(p.upper_split_ratio)))
            if p is None:
                raise ValueError(f'Expected calibration field in phase shifter {ps_tuple} but got None.')
            return pn.Column(
                hv.Overlay([
                    hv.Curve((vs_cal ** 2, p.upper_split_ratio), label='upper split'),
                    hv.Curve((vs_cal ** 2, p.lower_split_ratio), label='lower split'),
                    hv.Curve((vs_cal ** 2, p.split_ratio_fit), label='lower split fit').opts(
                        opts.Curve(line_dash='dashed')),
                    hv.Curve((vs_cal ** 2, p.upper_out), label='upper out'),
                    hv.Curve((vs_cal ** 2, p.lower_out), label='lower out'),
                    hv.Curve((vs_cal ** 2, p.upper_arm), label='upper arm'),
                    hv.Curve((vs_cal ** 2, p.lower_arm), label='lower arm'),
                    hv.Curve((vs_cal ** 2, p.total_arm), label='total arm'),
                    hv.Curve((vs_cal ** 2, p.total_out), label='total out')
                ]).opts(width=800, height=400, legend_position='right', shared_axes=False,
                        title='MZI Inspection Curves', xlabel='Electrical Power (Vsqr)', ylabel='Recorded Values'),
            )

        def abs_phase(phase: float):
            def f(*events):
                ps = tuple([int(c) for c in ps_dropdown.value.split(', ')])
                self.set_phase(ps, phase)

            return f

        def invert(*events):
            ps = tuple([int(c) for c in ps_dropdown.value.split(', ')])
            self.set_phase(ps, 2 * np.pi - self.ps[ps].phase)

        def rel_phase(phase_change: float):
            def f(*events):
                ps = tuple([int(c) for c in ps_dropdown.value.split(', ')])
                self.set_phase(ps, np.mod(self.ps[ps].phase + phase_change, 2 * np.pi))

            return f

        to_layer_button = pn.widgets.Button(name='To PS Layer')
        to_layer_button.on_click(to_layer)

        button_function_pairs = [
            (pn.widgets.Button(name='0', width=40), abs_phase(0)),
            (pn.widgets.Button(name='π/2', width=40), abs_phase(np.pi / 2)),
            (pn.widgets.Button(name='π', width=40), abs_phase(np.pi)),
            (pn.widgets.Button(name='3π/2', width=40), abs_phase(3 * np.pi / 2)),
            (pn.widgets.Button(name='-θ', width=40), invert),
            (pn.widgets.Button(name='+π', width=40), rel_phase(np.pi)),
            (pn.widgets.Button(name='-π', width=40), rel_phase(-np.pi)),
            (pn.widgets.Button(name='+π/2', width=40), rel_phase(np.pi / 2)),
            (pn.widgets.Button(name='-π/2', width=40), rel_phase(-np.pi / 2))
        ]

        for b, f in button_function_pairs:
            b.on_click(f)

        buttons = [b[0] for b in button_function_pairs]

        return pn.Column(
            ps_dropdown,
            pn.Row(*buttons),
            to_layer_button,
            calibration_image,
            calibrated_values,
        )

    def hessian_test(self, delta: float = 0.1):
        u = unitary_group.rvs(4)
        self.set_unitary(u)
        self.set_input(np.array((*u.T[-1], 0)))
        phase_shift_locs = ((5, 0), (7, 1), (9, 2), (4, 0), (6, 1), (8, 2))
        measurements = []
        for i in phase_shift_locs:
            for j in phase_shift_locs:
                self.ps[i].phase += delta
                self.ps[j].phase += delta
                time.sleep(0.1)
                measurements.append(self.fractional_right)
                self.ps[j].phase -= 2 * delta
                time.sleep(0.1)
                measurements.append(self.fractional_right)



    def get_unitary_phases(self):
        ts = [self.ps[tuple(t)].phase for t in self.network['theta_mesh']]
        ps = [self.ps[tuple(p)].phase for p in self.network['phi_mesh']]
        return ts, ps

    def set_unitary_phases(self, ts, ps):
        for t, th in zip(self.network['theta_mesh'], ts):
            self.ps[tuple(t)].phase = th
        for p, ph in zip(self.network['phi_mesh'], ps):
            self.ps[tuple(p)].phase = ph

    def to_calibration_file(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({ps.grid_loc: ps.calibration.dict for loc, ps in self.ps.items()}, f)

    def calibrate_all(self, pbar: Optional[Callable] = None):
        self.calibrate_thetas(pbar)
        self.calibrate_phis(pbar)

    def power_panel(self):
        def power_bars(data):
            return hv.Bars(data, hv.Dimension('Port'), 'Fractional power').opts(ylim=(0, 1))
        dmap = hv.DynamicMap(power_bars, streams=[self.spot_pipe]).opts(shared_axes=False)
        power_toggle = pn.widgets.Toggle(name='Power', value=False)
        lr_toggle = pn.widgets.Toggle(name='Left Spots', value=False)

        @gen.coroutine
        def update_plot():
            self.spot_pipe.send([(i, p) for i, p in enumerate(self.fractional_left
                                                              if lr_toggle.value else self.fractional_right)])
        cb = PeriodicCallback(update_plot, 100)

        def change_power(*events):
            for event in events:
                if event.name == 'value':
                    self.power_det_on = bool(event.new)
                    if self.power_det_on:
                        cb.start()
                    else:
                        cb.stop()

        power_toggle.param.watch(change_power, 'value')

        return pn.Column(dmap, power_toggle, lr_toggle)

    @property
    def fractional_right(self):
        return self.camera.spot_powers[::3] / np.sum(self.camera.spot_powers[::3])

    @property
    def fractional_left(self):
        return self.camera.spot_powers[2::3] / np.sum(self.camera.spot_powers[2::3])

    @property
    def fractional_center(self):
        return self.camera.spot_powers[1::3] / np.sum(self.camera.spot_powers[1::3])

    def default_panel(self):
        mesh_panel = self.mesh_panel()
        livestream_panel = self.camera.livestream_panel(cmap='gray')
        move_panel = self.stage.move_panel()
        power_panel = self.laser.power_panel()
        spot_panel = self.power_panel()
        wavelength_panel = self.laser.wavelength_panel()
        led_panel = self.led_panel()
        home_panel = self.home_panel()
        propagation_toggle_panel = self.propagation_toggle_panel()
        input_panel = self.input_panel()
        return pn.Column(
            pn.Pane(pn.Row(mesh_panel, input_panel), name='Mesh'),
            pn.Row(pn.Tabs(
                ('Live Interface',
                 pn.Column(
                     pn.Row(livestream_panel,
                            pn.Column(move_panel, home_panel, power_panel,
                                      wavelength_panel, led_panel,
                                      propagation_toggle_panel)
                            )
                 )),
                ('Calibration Panel', self.calibrate_panel())
            ), spot_panel)
        )

    def svd_proof(self, u, d, v, x):
        self.set_unitary(v.T)
        self.set_input(np.array((*x, 0)))
        time.sleep(0.05)
        res = np.sqrt(np.maximum(self.fractional_right[:4], 0)) * np.exp(1j * np.angle(v @ x)) * np.linalg.norm(x)
        res = res * d
        self.set_unitary(u.T)
        self.set_input(np.array((*res, 0)))
        time.sleep(0.05)
        return np.linalg.norm(res) * np.sqrt(np.maximum(self.fractional_right[:4], 0))


class PhaseShifter:
    def __init__(self, grid_loc: Tuple[int, int], spot_loc: Tuple[int, int],
                 voltage_channel: int, mesh: Sputnik,
                 meta_ps: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                 calibration: Optional[PhaseCalibration] = None):
        self.grid_loc = tuple(grid_loc)
        self.spot_loc = tuple(spot_loc)
        self.voltage_channel = voltage_channel
        self.meta_ps = [] if meta_ps is None else meta_ps
        self.mesh = mesh
        self.calibration = calibration
        self._phase = np.pi

    def calibrate(self, pbar: Optional[Callable] = None, vlim: Tuple[float, float] = (1, 5),
                  p0: Tuple[float, ...] = (1, 0, -0.01, 0.3, 0, 0), n_samples: int = 20000,
                  wait_time: float = 0):
        """Calibrate the phase shifter, setting calibration object to a new PhaseCalibration

        Args:
            pbar: Progress bar to keep track of the calibration time
            vlim: Voltage limits
            p0: Fit function initial value
            n_samples: Number of samples

        Returns:

        """

        layer, idx = self.spot_loc
        logger.info(f'Calibration of phase shifter {self.grid_loc} at spot {self.spot_loc}')
        for t in self.meta_ps:
            self.mesh.set_phase(t, np.pi / 2)
        vs, powers = self.mesh.sweep(self.voltage_channel, wait_time=wait_time,
                                     layer=layer, vlim=vlim, n_samples=n_samples, pbar=pbar)
        for t in self.meta_ps:
            self.mesh.set_phase(t, np.pi)
        try:
            self.calibration = PhaseCalibration(vs, powers, self.spot_loc, p0=p0)
        except RuntimeError:
            self.calibrate(pbar, vlim, p0, n_samples * 2, wait_time)

    def v2p(self, voltage: float):
        """Voltage to phase conversion for a give phase shifter

        Args:
            voltage: voltage to convert

        Returns:
            Phase converted from voltage

        """
        return self.calibration.v2p(voltage)

    def p2v(self, phase: float):
        """Phase to voltage conversion for a give phase shifter

        Args:
            phase: phase to convert

        Returns:
            Voltage converted from phase

        """
        return self.calibration.p2v(phase)

    @property
    def dict(self):
        return {
            'grid_loc': self.grid_loc,
            'spot_loc': self.spot_loc,
            'voltage_channel': self.voltage_channel,
            'meta_ps': self.meta_ps
        }

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: float):
        """Set the phase shifter in radians

        Args:
            phase: phase shift in the range [0, 2 * pi)

        Returns:

        """
        self.mesh.control.write_chan(self.voltage_channel, self.p2v(np.mod(phase, 2 * np.pi)))
        self._phase = phase

    def reset(self):
        """Reset the phase voltage to 0V and set the phase shift to np.nan

        Args:

        Returns:

        """
        # offset in temperature to reduce instability during calibration
        self.mesh.control.write_chan(self.voltage_channel, 2)
        self._phase = np.nan  # there might not be a phase defined here, so we just treat it as nan.

    def opt_spot(self, spot: Optional[Tuple[int, int]] = None, guess_phase: float = None,
                 wait_time: float = 0, n_samples: int = 100,
                 pbar: Optional[Callable] = None, move: bool = True, maximize: bool = False):
        """Maximize the power at a spot by sweeping the phase shift voltage

        Args:
            spot: Spot to minimize power
            wait_time: Wait time between samples
            n_samples: Number of samples
            pbar: Progress bar handle
            move: Whether to move the stage (to save time, set to false if the stage doesn't move)
            maximize: Whether to maximize the power or minimize the power

        Returns:

        """
        layer, idx = self.spot_loc if spot is None else spot
        if move:
            self.mesh.to_layer(layer)
        min_phase = 0 if guess_phase is None else np.maximum(0, guess_phase - 0.1)
        max_phase = 0 if guess_phase is None else np.minimum(2 * np.pi, guess_phase + 0.1)
        phases = np.linspace(0, 2 * np.pi, n_samples) if guess_phase is None else np.linspace(min_phase, max_phase)
        iterator = pbar(phases) if pbar is not None else phases
        self.phase = guess_phase
        time.sleep(1)
        powers = []
        for phase in iterator:
            self.phase = phase
            time.sleep(wait_time)
            p = self.mesh.camera.spot_powers
            powers.append(p[3 * idx] / (p[3 * idx] + p[3 * idx - 3]))
        opt_ps = phases[np.argmax(powers) if maximize else np.argmin(powers)]
        self.phase = opt_ps
        return opt_ps, powers
