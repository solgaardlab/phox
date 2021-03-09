from typing import Tuple, Callable, Optional, Dict

from .activephotonicsimager import ActivePhotonicsImager, _get_grating_spot

import time
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TriangularMeshImager(ActivePhotonicsImager):
    def __init__(self, interlayer_xy: Tuple[float, float], spot_xy: Tuple[int, int], interspot_xy: Tuple[int, int],
                 ps_assignments: Dict[Tuple[int, int], int], home: Tuple[float, float] = (0, 0),
                 stage_port: str = '/dev/ttyUSB1', laser_port: str = '/dev/ttyUSB0',
                 camera_calibration_filepath: Optional[str] = None, integration_time: int = 20000,
                 plim: Tuple[float, float] = (0.05, 4.25), vmax: float = 6):
        self.interlayer_xy = interlayer_xy
        self.spot_xy = spot_xy
        self.interspot_xy = interspot_xy
        self.ps_assignments = ps_assignments
        self.calibrations = {}
        super(TriangularMeshImager, self).__init__(home, stage_port, laser_port, camera_calibration_filepath,
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

    def mesh_sweep(self, ps_location: Tuple[int, int], layer: int, idx: int, vmax: float,
                   window_size: int = 15, wait_time: float = 0.2, resolution: int = 100,
                   pbar: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        self.to_layer(layer)
        vs = np.sqrt(np.linspace(0, vmax ** 2, resolution))
        centers = [(self.spot_xy[0] + self.interspot_xy[0] * idx, self.spot_xy[1]),
                   (self.spot_xy[0] + self.interspot_xy[0] * (idx + 1), self.spot_xy[1])]
        powers, _ = self.sweep_voltage(voltages=vs, centers=centers,
                                       channel=self.ps_assignments[ps_location],
                                       pbar=pbar, window_size=window_size, wait_time=wait_time)
        split_ratios = powers[1] / (powers[0] + powers[1])
        return vs, split_ratios
    #
    # def calibrate(self, ps_location: Tuple[int, int], layer: int, idx: int, vmax: float,
    #               window_size: int = 20, wait_time: float = 0.1, resolution: int = 100,
    #               pbar: Optional[Callable] = None):
    #     self.calibrations[ps_location] = self.mesh_sweep(ps_location, layer, idx, vmax,
    #                                                      win)

    def nullify(self, theta: Tuple[int, int], phi: Tuple[int, int], vmax: float = 5, resolution: int = 100,
                window_size: int = 15, pbar: Optional[Callable] = None):
        layer, idx = theta
        vs, phi_split_ratios = self.mesh_sweep(phi, layer - 1, idx, vmax, window_size=window_size,
                                               resolution=resolution, pbar=pbar)
        logger.info('Minimize using phi')
        opt_phi = vs[np.argmin(phi_split_ratios)]
        self.control.write_chan(self.ps_assignments[phi], opt_phi)
        time.sleep(1)
        logger.info('Minimize using theta')
        vs, theta_split_ratios = self.mesh_sweep(theta, layer - 1, idx, vmax,  window_size=window_size,
                                                 resolution=resolution, pbar=pbar)
        opt_theta = vs[np.argmin(theta_split_ratios)]
        self.control.write_chan(self.ps_assignments[theta], opt_theta)
        return opt_theta, opt_phi
