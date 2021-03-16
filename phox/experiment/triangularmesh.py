from typing import Tuple, Callable, Optional, Dict

from .activephotonicsimager import ActivePhotonicsImager, _get_grating_spot
from ..instrumentation import XCamera

import time
import numpy as np

import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TriangularMeshImager(ActivePhotonicsImager):
    def __init__(self, interlayer_xy: Tuple[float, float], spot_xy: Tuple[int, int], interspot_xy: Tuple[int, int],
                 ps_assignments: Dict[Tuple[int, int], int], window_shape: Tuple[int, int] = (15, 10),
                 home: Tuple[float, float] = (0, 0), stage_port: str = '/dev/ttyUSB1',
                 laser_port: str = '/dev/ttyUSB0', lmm_port: str = '/dev/ttyUSB2',
                 camera_calibration_filepath: Optional[str] = None, integration_time: int = 20000,
                 plim: Tuple[float, float] = (0.05, 4.25), vmax: float = 6):
        self.interlayer_xy = interlayer_xy
        self.spot_xy = s = spot_xy
        self.interspot_xy = ixy = interspot_xy
        self.ps_assignments = ps_assignments
        self.calibrations = {}
        self.spots = [(j * ixy[0] + s[0], i * ixy[1] + s[1], window_shape[0], window_shape[1])
                      for j in range(6) for i in range(3)]
        self.camera = XCamera(integration_time=integration_time, spots=self.spots)
        self.integration_time = integration_time
        super(TriangularMeshImager, self).__init__(home, stage_port, laser_port, lmm_port, camera_calibration_filepath,
                                                   integration_time, plim, vmax)

    def to_layer(self, layer: int):
        self.stage.move(x=self.home[0] + self.interlayer_xy[0] * layer,
                        y=self.home[1] + self.interlayer_xy[1] * layer)

    def mesh_img(self, n: int, wait_time: float = 2, window_size: int = 20):
        """

        Args:
            n: Number of inputs to the mesh
            wait_time: Time to wait for the stage to move to the next layer
            window_size: Window size for the spots

        Returns:

        """
        spots = []
        for m in range(n + 1):
            self.to_layer(3 * m)
            time.sleep(wait_time)
            img = self.camera.frame()
            s, ixy = self.spot_xy, self.interspot_xy
            if m < n:
                spots.append(np.hstack([np.vstack([_get_grating_spot(img, center=(j * ixy[0] + s[0], i * ixy[1] + s[1]),
                                                                     window_size=window_size)[1]
                                                   for j in range(n)]) for i in range(3)]))
            else:
                spots.append(np.vstack([_get_grating_spot(img, center=(j * ixy[0] + s[0], 2 * ixy[1] + s[1]),
                                                          window_size=window_size)[1] for j in range(n)]))
        return np.hstack(spots[::-1])

    def mesh_sweep(self, ps_location: Tuple[int, int], layer: int, vlim: Tuple[float, float],
                   wait_time: float = 0.1, n_samples: int = 1001,
                   pbar: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            ps_location: phase shifter location in the mesh
            layer: layer to move the stage
            vlim: Voltage limit for the sweep
            wait_time: Wait time between setting the temperature and taking the image
            n_samples: Number of samples
            pbar: progress bar (optional) to track the progress of the sweep

        Returns:

        """
        self.to_layer(layer)
        time.sleep(1)
        vs = np.sqrt(np.linspace(vlim[0] ** 2, vlim[1] ** 2, n_samples))
        iterator = pbar(vs) if pbar is not None else vs
        channel = self.ps_assignments[ps_location]
        powers = []
        for v in iterator:
            self.control.write_chan(channel, v)
            time.sleep(wait_time)
            powers.append(self.camera.spot_powers)
        return vs, np.asarray(powers).T

    def nullify(self, theta: Tuple[int, int], phi: Optional[Tuple[int, int]] = None,
                vlim: Tuple[float, float] = (0, 5), n_samples: int = 1001, wait_time: float = 0.01,
                pbar: Optional[Callable] = None, nullify_bottom: bool = True):
        layer, idx = theta
        idx_null, idx_max = (idx * 3, (idx + 1) * 3) if nullify_bottom else ((idx + 1) * 3, idx * 3)

        def nullify_ps(ps, name):
            vs, powers = self.mesh_sweep(ps, layer - 1, vlim=vlim, wait_time=wait_time,
                                         n_samples=n_samples, pbar=pbar)
            logger.info(f'Minimize power by adjusting {name} phase shifter at {ps}')
            split_ratios = powers[idx_null] / (powers[idx_max] + powers[idx_null])
            opt_ps = vs[np.argmin(split_ratios)]
            self.control.write_chan(self.ps_assignments[ps], opt_ps)
            return opt_ps

        if phi is not None:
            opt_phi = nullify_ps(phi, 'phi')
        else:
            opt_phi = None
        # time for temperature to settle
        time.sleep(1)
        opt_theta = nullify_ps(theta, 'phi')
        return opt_theta, opt_phi

    def nidaqmx_spot_callback(self):
        p = []

        def spot_update(task_idx, every_n_samples_event_type, num_of_samples, callback_data):
            p.append(self.camera.spot_powers)
            return 0
        return p, spot_update
