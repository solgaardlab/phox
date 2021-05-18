from typing import Tuple, Callable, Optional, Dict

from holoviews import opts
from holoviews.streams import Pipe

from .activephotonicsimager import ActivePhotonicsImager, _get_grating_spot
from ..instrumentation import XCamera
from dphox.demo import mesh

import time
import numpy as np

from ..model.meshsim import reck
from ..utils import vector_to_phases, phases_to_vector
import panel as pn
import pickle

import logging
import holoviews as hv
import hashlib

from ..model.phase import PhaseCalibration

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TriangularMeshImager(ActivePhotonicsImager):
    def __init__(self, interlayer_xy: Tuple[float, float], spot_xy: Tuple[int, int], interspot_xy: Tuple[int, int],
                 config: Dict, ps_calibration: Dict, window_shape: Tuple[int, int] = (15, 10),
                 backward_shift: float = 0.033, home: Tuple[float, float] = (0, 0), stage_port: str = '/dev/ttyUSB1',
                 laser_port: str = '/dev/ttyUSB0', lmm_port: str = '/dev/ttyUSB2',
                 camera_calibration_filepath: Optional[str] = None, integration_time: int = 20000,
                 plim: Tuple[float, float] = (0.05, 4.25), vmax: float = 6):
        """A class meant to specifically image 6 x 6 triangular mesh,
        but with the hope for generalization in the future.

        Args:
            interlayer_xy:
            spot_xy:
            interspot_xy:
            config: Configuration file for the circuit
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
        self.network = config['network']
        self.thetas = [PhaseShifter(**ps_dict, mesh=self,
                                    calibration=PhaseCalibration(**ps_calibration[tuple(ps_dict['grid_loc'])])
                                    ) for ps_dict in config['thetas']]
        self.thetas: Dict[Tuple[int, int], PhaseShifter] = {ps.grid_loc: ps for ps in self.thetas}
        self.phis = [PhaseShifter(**ps_dict, mesh=self,
                                  calibration=PhaseCalibration(**ps_calibration[tuple(ps_dict['grid_loc'])])
                                  ) for ps_dict in config['phis']]
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
        super(TriangularMeshImager, self).__init__(home, stage_port, laser_port, lmm_port, camera_calibration_filepath,
                                                   integration_time, plim, vmax)

        self.reset_control()
        self.set_transparent()
        self.camera.start_frame_loop()
        self.go_home()
        self.stage.wait_until_stopped()
        self.power_pipe = Pipe()
        self.ps_pipe = Pipe()
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
            self.backward = not self.backward
            self.control.ttl_toggle(chan)

        button = pn.widgets.Button(name='Switch Propagation Direction')
        button.on_click(toggle)
        return button

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
        button_list = [pn.widgets.Button(name=f'{i}', width=15) for i in range(5)]
        for i, button in enumerate(button_list):
            button.on_click(basis(i))
        buttons = pn.Row(*button_list)
        reset_button = pn.widgets.Button(name='Zero Voltages')
        reset_button.on_click(reset)
        return pn.Column(reset_button, bar_button, cross_button, alternating_button, uniform_button, buttons)

    def set_unitary(self, u):
        thetas, phis, _, phases = reck(u)
        for theta_ps, theta in zip(self.network['theta_mesh'], thetas[::-1]):
            self.ps[tuple(theta_ps)].phase = theta
        for phi_ps, phi in zip(self.network['phi_mesh'], phis[::-1]):
            self.ps[tuple(phi_ps)].phase = phi
        return phases

    def mesh_panel(self, power_cmap: str = 'hot', ps_cmap: str = 'greens'):
        polys = [np.asarray(p.exterior.coords.xy).T
                 for multipoly in mesh.path_array.flatten()
                 for p in multipoly]
        waveguides = hv.Polygons(polys).opts(data_aspect=1, frame_height=200,
                                             ylim=(-10, 70), xlim=(0, mesh.size[0]),
                                             color='black', line_width=2)
        phase_shift_polys = [np.asarray(p.buffer(1).exterior.coords.xy).T
                             for p in mesh.phase_shifter_array('m1am')[::3]]
        labels = np.fliplr(np.fliplr(np.mgrid[0:6, 0:19]).reshape((2, -1)).T)
        centroids = [(poly.centroid.x, poly.centroid.y) for poly in mesh.phase_shifter_array('m1am')[::3]]

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

    def calibrate_thetas(self, pbar: Optional[Callable] = None) -> Dict[Tuple[int, int], np.ndarray]:
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
            ps.calibrate(pbar)
            ps.phase = np.pi

    def calibrate_phis(self, pbar: Optional[Callable] = None):
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
            ps.calibrate(pbar)
            ps.phase = 0

    def set_input(self, vector: np.ndarray, add_normalization: bool = False, theta_only: bool = False):
        n = 4
        vector = vector.conj()
        if add_normalization:
            vector = vector / np.sqrt(np.sum(np.abs(vector))) * np.sqrt(n / (n + 1))
            vector = np.append(vector, np.sqrt(1 / (n + 1)))
        # design inconsistency on chip led to this:
        lower_phi = (0, 0, 1, 0, 0) if self.backward else (0, 0, 0, 0, 0)
        lower_theta = (0, 0, 0, 0, 0)

        thetas, phis = vector_to_phases(vector, lower_theta, lower_phi)

        for i in range(n):
            phase = {'theta': thetas[i], 'phi': phis[i]}
            for var in ('theta',) if theta_only else ('theta', 'phi'):
                key = f'{var}_right' if self.backward else f'{var}_left'
                self.set_phase(self.network[key][i], phase[var])
        self.set_phase(self.network['theta_ref'], np.pi)

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
            phi_vs.append(self.ps[phi].opt_spot(spot=(16, 3 - i), n_samples=501, wait_time=0.01, move=False)[0])
            theta_vs.append(self.ps[theta].opt_spot(spot=(16, 3 - i), n_samples=501, wait_time=0.01, move=False)[0])
            if update_mesh:
                self.update_mesh_image()

        thetas = np.asarray(theta_vs)  # change this based on calibration
        phis = np.asarray(phi_vs)  # change this based on calibration

        return phases_to_vector(thetas, phis, lower_theta, lower_phi)

    def calibrate_panel(self):
        vs = np.sqrt(np.linspace(2 ** 2, 5.5 ** 2, 10000))
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

        @pn.depends(ps_dropdown.param.value)
        def calibration_image(value):
            ps_tuple = tuple([int(c) for c in value.split(', ')])
            p = self.ps[ps_tuple].calibration
            if p is None:
                raise ValueError(f'Expected calibration field in phase shifter {ps_tuple} but got None.')
            return pn.Column(
                hv.Overlay([
                    hv.Curve((vs ** 2, p.upper_split_ratio), label='upper split'),
                    hv.Curve((vs ** 2, p.lower_split_ratio), label='lower split'),
                    hv.Curve((vs ** 2, p.split_ratio_fit), label='lower split fit').opts(opts.Curve(line_dash='dashed')),
                    hv.Curve((vs ** 2, p.upper_out), label='upper out'),
                    hv.Curve((vs ** 2, p.lower_out), label='lower out'),
                    hv.Curve((vs ** 2, p.upper_arm), label='upper arm'),
                    hv.Curve((vs ** 2, p.lower_arm), label='lower arm'),
                    hv.Curve((vs ** 2, p.total_arm), label='total arm'),
                    hv.Curve((vs ** 2, p.total_out), label='total out')
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
            calibration_image,
            calibrated_values,
        )

    def get_opow_hash(self, input_obj):
        hasher1 = hashlib.sha3_256()
        hasher1.update(input_obj)
        x = list(np.zeros(64))
        for i in range(0, 64, 2):
            j = i >> 1
            x[i] = hasher1.digest()[j] >> 4
            x[i + 1] = hasher1.digest()[j] & 0x0F
        y = []
        for i in range(len(x)):
            y.append(0)
            for j in range(len(x)):
                y[i] += M[i][j] * x[j]
            y[i] = y[i] >> 10
        preout = bytearray()
        for i in range(32):
            a = y[i << 1]
            b = y[(i << 1) + 1]
            preout.append(((a << 4) | b) ^ hasher1.digest()[i])
        hasher2 = hashlib.sha3_256()
        hasher2.update(preout)
        final_hash = hasher2.digest()
        return final_hash

    def to_calibration_file(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({ps.grid_loc: ps.calibration.dict for loc, ps in self.ps.items()}, f)

    def calibrate_all(self, pbar: Optional[Callable] = None):
        self.calibrate_thetas(pbar)
        self.calibrate_phis(pbar)


class PhaseShifter:
    def __init__(self, grid_loc: Tuple[int, int], spot_loc: Tuple[int, int],
                 voltage_channel: int, mesh: TriangularMeshImager,
                 meta_ps: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                 calibration: Optional[PhaseCalibration] = None):
        self.grid_loc = tuple(grid_loc)
        self.spot_loc = tuple(spot_loc)
        self.voltage_channel = voltage_channel
        self.meta_ps = [] if meta_ps is None else meta_ps
        self.mesh = mesh
        self.calibration = calibration
        self._phase = np.pi

    def calibrate(self, pbar: Optional[Callable] = None, vlim: Tuple[float, float] = (2, 5.5),
                  p0: Tuple[float, ...] = (1, 0, 0, 0.3, 0, 0), n_samples: int = 10000):
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
        vs, powers = self.mesh.sweep(self.voltage_channel,
                                     layer, vlim=vlim, n_samples=n_samples, pbar=pbar)
        for t in self.meta_ps:
            self.mesh.set_phase(t, np.pi)
        self.calibration = PhaseCalibration(vs, powers, self.spot_loc, p0=p0)

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
        self.mesh.control.write_chan(self.voltage_channel, self.p2v(phase))
        self._phase = phase

    def reset(self):
        """Reset the phase voltage to 0V and set the phase shift to np.nan

        Args:

        Returns:

        """
        # offset in temperature to reduce instability during calibration
        self.mesh.control.write_chan(self.voltage_channel, 2)
        self._phase = np.nan  # there might not be a phase defined here, so we just treat it as nan.

    def opt_spot(self, spot: Optional[Tuple[int, int]] = None,
                 wait_time: float = 0, n_samples: int = 1000,
                 pbar: Optional[Callable] = None, move: bool = True):
        """Maximize the power at a spot by sweeping the phase shift voltage

        Args:
            spot: Spot to minimize power
            wait_time: Wait time between samples
            n_samples: Number of samples
            pbar: Progress bar handle
            move: Whether to move the stage (to save time, set to false if the stage doesn't move)

        Returns:

        """
        layer, idx = self.spot_loc if spot is None else spot
        if move:
            self.mesh.to_layer(layer)
        max_phase = 2 * np.pi if len(self.meta_ps) == 2 else np.pi  # cheat to distinguish between theta and phi

        phases = np.linspace(0, max_phase, n_samples)
        iterator = pbar(phases) if pbar is not None else phases
        self.phase = 0
        time.sleep(1)
        powers = []
        for phase in iterator:
            self.phase = phase
            time.sleep(wait_time)
            p = self.mesh.camera.spot_powers
            powers.append(p[3 * idx] / (p[3 * idx] + p[3 * idx + 3]))
        opt_ps = phases[np.argmax(powers)]
        self.phase = opt_ps
        return opt_ps, powers

        # vlim = (self.p2v(0), self.p2v(max_phase)) if vlim is None else vlim
        # vs, powers = self.mesh.sweep(self.voltage_channel,
        #                              layer - 1, vlim=vlim, wait_time=wait_time,
        #                              n_samples=n_samples, pbar=pbar, move=move)
        # logger.info(f'Minimize power for phase shifter {self.grid_loc}, at spot {spot}')
        # split_ratios = powers[idx, 0]
        # opt_ps = vs[np.argmin(split_ratios)]
        #
        # self.mesh.control.write_chan(self.voltage_channel, opt_ps)
        # return opt_ps
