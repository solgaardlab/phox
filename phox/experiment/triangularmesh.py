from typing import Tuple, Callable, Optional, Dict, List

from holoviews.streams import Pipe

from .activephotonicsimager import ActivePhotonicsImager, _get_grating_spot
from ..instrumentation import XCamera
from dphox.demo import mesh

import time
import numpy as np
from scipy.optimize import curve_fit
from ..utils import vector_to_phases, phases_to_vector, cal_v_power, cal_phase_v
import panel as pn

import logging
import holoviews as hv

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TriangularMeshImager(ActivePhotonicsImager):
    def __init__(self, interlayer_xy: Tuple[float, float], spot_xy: Tuple[int, int], interspot_xy: Tuple[int, int],
                 config: Dict, n: int = 4, window_shape: Tuple[int, int] = (15, 10), backward_shift: float = 0.033,
                 home: Tuple[float, float] = (0, 0), stage_port: str = '/dev/ttyUSB1',
                 laser_port: str = '/dev/ttyUSB0', lmm_port: str = '/dev/ttyUSB2',
                 camera_calibration_filepath: Optional[str] = None, integration_time: int = 20000,
                 plim: Tuple[float, float] = (0.05, 4.25), vmax: float = 6):
        self.n = n
        self.interlayer_xy = interlayer_xy
        self.spot_xy = s = spot_xy
        self.interspot_xy = ixy = interspot_xy

        self.spots = [(j * ixy[0] + s[0], i * ixy[1] + s[1], window_shape[0], window_shape[1])
                      for j in range(6) for i in range(3)]
        self.camera = XCamera(integration_time=integration_time, spots=self.spots)
        self.integration_time = integration_time
        self.ps_to_channel = config['ps_to_channel']
        self.ps_to_mesh = config['ps_to_mesh']
        self.ps_calibrations = config['ps_calibrations']
        self.theta_ps = config['thetas']
        self.phi_ps = config['phis']
        self.phi_cal_cfg = config['phi_calibrations']
        self.theta_cal_cfg = config['theta_calibrations']
        self.backward = False
        self.backward_shift = backward_shift
        super(TriangularMeshImager, self).__init__(home, stage_port, laser_port, lmm_port, camera_calibration_filepath,
                                                   integration_time, plim, vmax)

        self.reset_control()
        self.set_transparent()
        self.camera.start_frame_loop()
        self.go_home()
        self.stage.wait_until_stopped()
        self.mesh_pipe = Pipe()
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
                powers.append(np.hstack([np.vstack([_get_grating_spot(img, center=(j * ixy[0] + s[0], i * ixy[1] + s[1]),
                                                                      window_size=window_size)[0]
                                                   for j in range(n)]) for i in range(3)]))
                spots.append(np.hstack([np.vstack([_get_grating_spot(img, center=(j * ixy[0] + s[0], i * ixy[1] + s[1]),
                                                                     window_size=window_size)[1] / np.sum(powers[-1][:, i])
                                                   for j in range(n)]) for i in range(3)]))
            else:
                powers.append(np.vstack([_get_grating_spot(img, center=(j * ixy[0] + s[0], s[1]),
                                                           window_size=window_size)[0] for j in range(n)]))
                spots.append(np.vstack([_get_grating_spot(img, center=(j * ixy[0] + s[0], s[1]),
                                                          window_size=window_size)[1] / np.sum(powers[-1])
                                        for j in range(n)]))
        return np.fliplr(np.hstack(powers[::-1])), np.fliplr(np.hstack(spots[::-1]))

    def mesh_sweep(self, ps_location: Tuple[int, int], layer: int, vlim: Tuple[float, float],
                   wait_time: float = 0, n_samples: int = 1001, move: bool = True,
                   pbar: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            ps_location: phase shifter location in the mesh
            layer: layer to move the stage
            vlim: Voltage limit for the sweep
            wait_time: Wait time between setting the temperature and taking the image
            n_samples: Number of samples
            move: whether to move to the appropriate layer (mainly useful for output nullify function)
            pbar: progress bar (optional) to track the progress of the sweep

        Returns:

        """
        if move:
            self.to_layer(layer)
        vs = np.sqrt(np.linspace(vlim[0] ** 2, vlim[1] ** 2, n_samples))
        iterator = pbar(vs) if pbar is not None else vs
        channel = self.ps_to_channel[ps_location]
        powers = []
        for v in iterator:
            self.control.write_chan(channel, v)
            time.sleep(wait_time)
            powers.append(self.camera.spot_powers)
        return vs, np.asarray(powers).T

    def reset_control(self):
        for device in self.control.system.devices:
            device.reset_device()

    def nullify(self, theta: Tuple[int, int] = None, phi: Optional[Tuple[int, int]] = None,
                spot: Tuple[int, int] = None, move: bool = True,
                vlim_theta: Tuple[float, float] = None, vlim_phi: Tuple[float, float] = None,
                n_samples: int = 1001, wait_time: float = 0.01,
                pbar: Optional[Callable] = None, lower_theta: bool = False, bottom: bool = True):
        layer, idx = spot
        if not lower_theta:
            idx -= 1  # ensures if theta PS is the top of the MZI
        idx_null, idx_max = (idx * 3, (idx + 1) * 3) if bottom else ((idx + 1) * 3, idx * 3)

        vlim = {
            # 'theta': (self.p2v(theta, 0), self.p2v(theta, 2 * np.pi)) if vlim_theta is None else vlim_theta,
            'phi': (self.p2v(phi, 0), self.p2v(phi, 2 * np.pi)) if vlim_phi is None else vlim_phi
        }

        def nullify_ps(ps, name):
            vs, powers = self.mesh_sweep(ps, layer - 1, vlim=vlim[name], wait_time=wait_time,
                                         n_samples=n_samples, pbar=pbar, move=move)
            logger.info(f'Minimize power by adjusting {name} phase shifter at {ps}')
            split_ratios = powers[idx_null] / (powers[idx_max] + powers[idx_null])
            opt_ps = vs[np.argmin(split_ratios)]
            self.control.write_chan(self.ps_to_channel[ps], opt_ps)
            return opt_ps

        if phi is not None:
            opt_phi = nullify_ps(phi, 'phi')
            time.sleep(0.5)
        else:
            opt_phi = None

        if theta is not None:
            # time for temperature to settle
            opt_theta = nullify_ps(theta, 'theta')
            time.sleep(0.5)
        else:
            opt_theta = None
        return opt_theta, opt_phi

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

    def transparent_button_panel(self):
        def transparent_bar(*events):
            self.set_transparent()

        def transparent_cross(*events):
            self.set_transparent(bar=False)

        bar_button = pn.widgets.Button(name='Transparent (Bar)')
        bar_button.on_click(transparent_bar)
        cross_button = pn.widgets.Button(name='Transparent (Cross)')
        cross_button.on_click(transparent_cross)
        return pn.Column(bar_button, cross_button)

    def nidaqmx_spot_callback(self):
        p = []

        def spot_update(task_idx, every_n_samples_event_type, num_of_samples, callback_data):
            p.append(self.camera.spot_powers)
            return 0

        return p, spot_update

    def mesh_panel(self, cmap: str = 'hot'):
        polys = [np.asarray(p.exterior.coords.xy).T
                 for multipoly in mesh.path_array.flatten()
                 for p in multipoly]
        waveguides = hv.Polygons(polys).opts(data_aspect=1, frame_height=200,
                                             ylim=(-10, 70), xlim=(0, mesh.size[0]),
                                             color='black', line_width=2)
        colored_polys = lambda data: hv.Polygons(
            [{('x', 'y'): poly, 'power': z} for poly, z in zip(polys, data)], vdims='power'
        )
        powers_mesh = hv.DynamicMap(colored_polys, streams=[self.mesh_pipe]).opts(
            data_aspect=1, frame_height=200, ylim=(-10, 70),
            xlim=(0, mesh.size[0]), line_color='none', cmap=cmap, shared_axes=False
        )
        self.mesh_pipe.send(np.full(len(polys), np.nan))
        image_button = pn.widgets.Button(name='Mesh Image')
        def mesh_image(*events):
            powers, spots = self.mesh_img(self.n + 2)
            powers[powers <= 0] = 0
            powers = np.flipud(np.sqrt(powers / np.max(powers)))
            self.mesh_pipe.send([p for p, multipoly in zip(powers.flatten(), mesh.path_array.flatten())
                                 for _ in multipoly])
        image_button.on_click(mesh_image)
        return pn.Column(waveguides * powers_mesh, image_button)

    def calibrate(self, ps: Tuple[int, int], spot: Tuple[int, int] = None, lower: bool = False, vmax: float = 5.5,
                  n_samples: int = 10000, p0: Tuple[float, ...] = (1, 0, 0, 0.3, 0, 0),
                  pbar: Optional[Callable] = None):
        if spot is None:
            layer, idx = ps
            if not lower:
                idx -= 1  # ensures if theta PS is the top of the MZI
            vs, powers = self.mesh_sweep(ps, layer - 1, vlim=(0, vmax), n_samples=n_samples, pbar=pbar)
        else:
            layer, idx = spot
            vs, powers = self.mesh_sweep(ps, layer, vlim=(0, vmax), n_samples=n_samples, pbar=pbar)
        idx_t, idx_r = (idx * 3, (idx + 1) * 3)
        ts = powers[idx_t] / (powers[idx_t] + powers[idx_r])
        p, _ = curve_fit(cal_v_power, vs, ts, p0=p0)
        q, _ = curve_fit(cal_phase_v, np.polyval(p[2:], vs), vs ** 2)
        self.ps_calibrations[ps] = np.vstack([p[2:], q])
        return vs, powers

    def set_transparent(self, bar: bool = True):
        for ps in self.ps_calibrations:
            self.set_phase(ps, np.pi if bar else 0)

    def set_phase(self, ps: Tuple[int, int], phase: float):
        """Set the phase shifter in the mesh using raw phase value

        Args:
            ps: phase shifter grid location in the mesh
            phase: phase to convert in the range [0, 2 * pi)

        Returns:

        """
        # TODO(sunil): fix this calibration!
        phase = np.mod(np.pi - phase, 2 * np.pi) if ps == (13, 3) else phase
        p = phase / 2 + np.pi
        self.control.write_chan(self.ps_to_channel[ps], self.p2v(ps, p))

    def v2p(self, ps: Tuple[int, int], voltage: float):
        """Voltage to phase conversion for a give phase shifter

        Args:
            ps: phase shifter grid location in the mesh
            voltage: voltage to convert

        Returns:
            Phase converted from voltage

        """
        p, _ = self.ps_calibrations[ps]
        return np.polyval(p, voltage)

    def p2v(self, ps: Tuple[int, int], phase: float):
        """Phase to voltage conversion for a give phase shifter

        Args:
            ps: phase shifter grid location in the mesh
            phase: phase to convert

        Returns:
            Voltage converted from phase

        """
        _, q = self.ps_calibrations[ps]
        return np.sqrt(np.abs(np.polyval(q, phase)))  # abs suppresses in case negative value

    def calibrate_thetas(self, pbar: Optional[Callable] = None) -> Dict[Tuple[int, int], np.ndarray]:
        """Row-wise calibration of the :math:`\\theta` phase shifters

        Args:
            pbar: Progress bar to keep track of each calibration

        Returns:
            Voltages used for the calibration and the resulting powers

        """
        idx = 0
        powers = {}
        self.set_transparent()

        for thetas in pbar(self.theta_cal_cfg):
            idx += 1
            for ps in pbar(thetas):
                _, p = self.calibrate(ps, pbar=pbar, lower=ps[1] < idx,
                                      p0=(1, 0, 0, 0.3, 0, -np.pi / 2) if ps == (15, 2) else (1, 0, 0, 0.3, 0, 0))
                powers[ps] = p
                self.set_phase(ps, np.pi)  # bar state
            self.set_phase(thetas[0], 0)  # cross state
        return powers

    def calibrate_phis(self, pbar: Optional[Callable] = None) -> Dict[Tuple[int, int], np.ndarray]:
        self.set_transparent()
        for t in self.ps_to_mesh['theta_left']:
            self.set_phase(t, 0)
        powers = {}
        for i, meta_mzis in enumerate(self.phi_cal_cfg):
            self.set_phase(self.ps_to_mesh['theta_left'][3 - i], np.pi)
            for spot, thetas_phis in meta_mzis.items():
                thetas, phis = thetas_phis['thetas'], thetas_phis['phis']
                # meta-MZI
                for t in thetas:
                    self.set_phase(t, np.pi / 2)
                for ps in phis:
                    print(ps)
                    _, p = self.calibrate(ps, pbar=pbar, spot=spot,
                                          p0=(1, 0, 0, 0.3, 0, -np.pi / 2) if ps == (14, 0) else (1, 0, 0, 0.3, 0, 0))
                    powers[ps] = p
                    self.set_phase(ps, 0)
                for t in thetas:
                    self.set_phase(t, np.pi)
        return powers

    def set_input(self, vector: np.ndarray, forward: bool = True):
        n = self.n
        vector = vector.conj() / np.sum(np.abs(vector)) * n / (n + 1)
        vector = np.append(vector, 1 / (n + 1))
        # design inconsistency on chip led to this:
        lower_phi = (0, 0, 0, 0, 0) if forward else (0, 0, 1, 0, 0)
        lower_theta = (0, 0, 0, 0, 0)

        thetas, phis = vector_to_phases(vector, lower_theta, lower_phi)

        for i in range(n):
            phase = {'theta': thetas[i], 'phi': phis[i]}
            for var in ('theta', 'phi'):
                key = f'{var}_left' if forward else f'{var}_right'
                ps = self.ps_to_mesh[key][i]
                self.set_phase(ps, phase[var])

    def read_output(self, forward: bool = True, pbar: Optional[Callable] = None):
        direction = 'right' if forward else 'true'
        theta_ps = self.ps_to_mesh[f'theta_{direction}'][::-1]
        phi_ps = self.ps_to_mesh[f'phi_{direction}'][::-1]

        theta_vs, phi_vs = [], []

        lower_phi = (0, 0, 0, 0, 0) if forward else (0, 0, 1, 0, 0)
        lower_theta = (0, 0, 0, 0, 0)

        self.to_layer(self.n * 4 if forward else 0)

        # TODO(sunil): fix hardcoding
        for i in range(self.n):
            v_theta, v_phi = self.nullify(theta=theta_ps[i], phi=phi_ps[i], move=False,
                                          n_samples=351, wait_time=0.01, pbar=pbar, bottom=False)
            theta_vs.append(self.v2p(theta_ps[i], v_theta))
            phi_vs.append(self.v2p(phi_ps[i], v_phi))

        thetas = np.asarray(theta_vs)  # change this based on calibration
        phis = np.asarray(phi_vs)  # change this based on calibration

        return phases_to_vector(thetas, phis, lower_theta, lower_phi)

