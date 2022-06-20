from ..instrumentation import ASI, NIDAQControl, XCamera, LaserHP8164A, LightwaveMultimeterHP8163A
from typing import Tuple, Callable, Optional, List
import numpy as np
import time

from skimage import measure
from shapely.geometry import Polygon

import logging
logger = logging.getLogger()
logger.setLevel(logging.WARN)


class ActivePhotonicsImager:
    def __init__(self, home: Tuple[float, float] = (0, 0),
                 stage_port: str = '/dev/ttyUSB0', laser_port: str = '/dev/ttyUSB1', lmm_port: str = '/dev/ttyUSB2',
                 camera_calibration_filepath: Optional[str] = None, integration_time: int = 20000,
                 plim: Tuple[float, float] = (0.05, 4.25), vmax: float = 6):
        """Active photonics imager, incorporating stage, camera livestream, and voltage control.

        Args:
            home: Home position for the stage
            stage_port: Stage serial port str
            laser_port: Laser serial port str
            lmm_port: Laser multimeter serial port str
            camera_calibration_filepath: Camera calibration file (for Xenics bobcat camera)
            integration_time: Integration time for the camera
            plim: Allowed laser power limits (min, max)
            vmax: Maximum allowed voltage
        """
        self.home = home
        logger.info('Connecting to camera...')
        if 'camera' not in self.__dict__:
            self.camera = XCamera(integration_time=integration_time)
            self.integration_time = integration_time
        self.camera.start()
        if camera_calibration_filepath is not None:
            self.camera.load_calibration(camera_calibration_filepath)
        logger.info('Connecting to stage...')
        self.stage = ASI(port=stage_port)
        self.stage.connect()
        logger.info('Connecting to mesh voltage control...')
        self.control = NIDAQControl(0, vmax)
        logger.info('Connecting to laser...')
        self.laser = LaserHP8164A(port=laser_port)
        self.laser.connect()
        logger.info('Connecting to lightwave multimeter')
        if lmm_port is not None:
            self.lmm = LightwaveMultimeterHP8163A(port=lmm_port)
            self.lmm.connect()
        logger.info('Turning laser off...')
        self.laser.state = 0
        time.sleep(1)
        logger.info('Taking camera reference...')
        self.camera.background_reference = self.camera.frame()
        time.sleep(1)
        logger.info('Turning laser back on...')
        self.laser.state = 1
        self.plim = plim
        self.vmax = vmax

    def go_home(self):
        self.stage.move(*self.home)

    def sweep_voltage(self, voltages: np.ndarray, centers: List[Tuple[int, int]], channel: int, pbar: Callable,
                      window_size: int, wait_time: float = 1,
                      integration_time: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Sweep a phase shifter voltage and monitor the output spots at specified centers for each voltage.

        Args:
            voltages: Voltages
            centers: Centers for which to extract grating spots (power and window)
            channel: Control channel to sweep
            pbar: Progress bar
            window_size: Window size
            wait_time: Wait time between voltage setting and image
            integration_time: Integration time for the sweep (default to current camera integration time)

        Returns:
            A tuple of powers and windows containing the spots

        """
        spots = []
        self.camera.set_integration_time(self.integration_time if integration_time is None else integration_time)
        iterator = pbar(voltages) if pbar is not None else voltages
        for v in iterator:
            self.control.write_chan(channel, v)
            time.sleep(wait_time)
            img = self.camera.frame()
            spots.append([_get_grating_spot(img, center, window_size) for center in centers])
        return np.asarray([[s[c][0] for s in spots] for c in range(len(centers))]),\
               np.asarray([[s[c][1] for s in spots] for c in range(len(centers))])

    def centers(self, threshold: int = 0) -> List[Tuple[int, int]]:
        """Determine the input and output centers for the MZI (avoids needing to hard-code positions)

        Args:
            threshold: Threshold

        Returns:
            a list of center pixel locations

        """
        self.camera.set_integration_time(self.integration_time)
        img = self.camera.frame()
        time.sleep(0.2)
        contours = [Polygon(np.fliplr(contour))
                    for contour in measure.find_contours(img, threshold) if len(contour) > 3]
        contours = [contour for contour in contours if contour.area > 2]
        contour_centers = [(int(contour.centroid.y), int(contour.centroid.x)) for contour in contours]
        return contour_centers

    def spot_saturation(self, center: Tuple[int, int], window_size: int = 10,
                        plim: Tuple[float, float] = (0.05, 4.25), n_steps: int = 421,
                        pbar: Optional[Callable] = None, wait_time: float = 1,
                        init_wait_time: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Measure spot saturation in a given grating spot (should be done for each grating spot!)

        Args:
            center: (x, y) for the center of the calibration image
            window_size: window size in the calibration image to compute the power
            plim: power range (pmin, pmax) for the calibration
            n_steps: number of steps in the sweep (resolution)
            pbar: callable for progressbar (for longer calibrations)
            wait_time: integration time for the laser power calibration
            init_wait_time: initial wait time

        Returns:
            A tuple of laser powers and measured powers associated with the calibration

        """
        laser_powers = np.linspace(*plim, n_steps)
        measured_powers = []
        measured_windows = []
        self.camera.set_integration_time(self.integration_time)
        iterator = pbar(laser_powers) if pbar is not None else laser_powers
        self.laser.power = plim[0]
        time.sleep(init_wait_time)
        for power in iterator:
            self.laser.power = power
            time.sleep(wait_time)
            img = self.camera.frame()
            power, window = _get_grating_spot(img, center, window_size)
            measured_powers.append(power)
            measured_windows.append(window)
        return laser_powers, np.asarray(measured_powers), np.stack(measured_windows)

    def dispersion(self, centers: List[Tuple[int, int]], window_size: int = 20,
                   wlim: Tuple[float, float] = (1.53, 1.57), n_steps: int = 401,
                   integration_time: int = 4000, pbar: Optional[Callable] = None, wait_time: float = 1,
                   init_wait_time: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Determine the dispersion relationship

        Args:
            centers: list of (x, y) for the center of the dispersion images
            window_size: window size for spots in the image to compute the power
            wlim: wavelength range (wmin, wmax) for the calibration
            n_steps: number of steps in the sweep (resolution)
            n_avg: number of times to average to determine the power
            integration_time: integration time for the laser power calibration
            pbar: callable for progressbar (for longer calibrations)

        Returns:
            A tuple of wavelengths, and measured powers associated with the calibration

        """
        wavelengths = np.linspace(*wlim, n_steps)
        measured_powers = []
        measured_windows = []
        self.camera.set_integration_time(integration_time)
        iterator = pbar(wavelengths) if pbar is not None else wavelengths
        self.laser.wavelength = wlim[0]
        time.sleep(init_wait_time)
        for wavelength in iterator:
            self.laser.wavelength = wavelength
            time.sleep(wait_time)
            img = self.camera.frame()
            res = [_get_grating_spot(img, center, window_size) for center in centers]
            power, window = [r[0] for r in res], [r[1] for r in res]
            measured_powers.append(np.asarray(power))
            measured_windows.append(np.asarray(window))
        return wavelengths, np.asarray(measured_powers), np.asarray(measured_windows)

    def shutdown(self):
        """Stop the camera and close the stage.

        """
        self.camera.stop()
        self.stage.close()



def _get_grating_spot(img: np.ndarray, center: Tuple[int, int], window_dim: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    window = img[center[0] - window_dim[0] // 2:center[0] - window_dim[0] // 2 + window_dim[0],
                 center[1] - window_dim[1] // 2:center[1] - window_dim[1] // 2 + window_dim[1]]
    power = np.sum(window)
    return power, window
