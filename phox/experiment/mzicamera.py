from ..instrumentation import ASI, MeshAOControl, XCamera, LaserHP8164A
from typing import Tuple, Callable, Optional
import numpy as np
import time

from skimage import measure
from shapely.geometry import Polygon


class MZICamera:
    def __init__(self, home: Tuple[float, float], dy: float):
        self.camera = XCamera()
        self.camera.start()
        self.stage = ASI()
        self.stage.connect()
        self.control = MeshAOControl()
        self.laser = LaserHP8164A()
        self.laser.connect()
        self.home = home
        self.x_home, self.y_home = home
        self.dy = dy

    def go_home(self):
        self.stage.move(*self.home)

    def current_image(self, n: int, max_invert: float = 27000):
        imgs = np.asarray([self.camera.frame() for _ in range(n)])
        return max_invert - np.mean(imgs, axis=0)

    def mesh_pics(self, n: int, max_invert: float = 27000):
        mesh_pics = []
        for m in range(7):
            layer = 1 + m * 3
            self.stage.move(x=self.x_home, y=self.y_home + layer * self.dy)
            time.sleep(3)
            mesh_pics.append(self.current_image(n, max_invert))
        return mesh_pics

    def mzi_sweep(self, voltages: np.ndarray, channel: int, spot_extract_integration_time: int, threshold: int,
                  ps_sweep_integration_time: int, pbar: Callable, n: int, window_size: int):
        output_powers = []
        input_powers = []
        input_center, output_center = self.io_centers(threshold, n, spot_extract_integration_time)
        self.camera.set_integration_time(ps_sweep_integration_time)
        for v in pbar(voltages):
            self.control.write_chan(channel, v)
            time.sleep(0.2)
            img = self.current_image(n)
            input_power, _ = _get_grating_spot(img, input_center, window_size)
            output_power, _ = _get_grating_spot(img, output_center, window_size)
            output_powers.append(output_power)
            input_powers.append(input_power)
        return input_powers, output_powers

    def centers(self, integration_time: int = 4000, threshold: int = 10000, n: int = 50):
        """Determine the input and output centers for the MZI (avoids needing to hard-code positions)

        Args:
            integration_time: Integration time
            threshold: Threshold
            n: Number of images to average to determine count

        Returns:

        """
        self.camera.set_integration_time(integration_time)
        img = self.current_image(n)
        time.sleep(0.2)
        contours = [Polygon(np.fliplr(contour)) for contour in measure.find_contours(img, threshold) if
                    len(contour) > 3]
        contours = [contour for contour in contours if contour.area > 2]
        contour_centers = [(int(contour.centroid.y), int(contour.centroid.x)) for contour in contours]
        return contour_centers

    def center_spot(self, center: Tuple[int, int], n: int, window_size: int):
        img = self.current_image(n)
        return _get_grating_spot(img, center, window_size)

    def calibrate_power(self, center: Tuple[int, int], window_size: int = 10,
                        min_power: float = 0.05, max_power: float = 4.25, n_steps: int = 421, n_avg: int = 50,
                        integration_time: int = 4000, pbar: Optional[Callable] = None, wait_time: float = 1,
                        init_wait_time: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calibrate laser power to the spot measurements in a given grating spot
           (should be done for each grating spot!)

        Args:
            center: (x, y) for the center of the calibration image
            window_size: window size in the calibration image to compute the power
            min_power: minimum power in the sweep
            max_power: maximum power in the sweep
            n_steps: number of steps in the sweep (resolution)
            n_avg: number of times to average to determine the power
            integration_time: integration time for the laser power calibration
            pbar: callable for progressbar (for longer calibrations)

        Returns:
            A tuple of laser powers and measured powers associated with the calibration

        """
        laser_powers = np.linspace(min_power, max_power, n_steps)
        measured_powers = []
        measured_windows = []
        self.camera.set_integration_time(integration_time)
        iterator = pbar(laser_powers) if pbar is not None else laser_powers
        self.laser.set_power(min_power)
        time.sleep(init_wait_time)
        for power in iterator:
            self.laser.set_power(power)
            time.sleep(wait_time)
            img = self.current_image(n_avg)
            power, window = _get_grating_spot(img, center, window_size)
            measured_powers.append(power)
            measured_windows.append(window)
        return laser_powers, np.asarray(measured_powers), np.stack(measured_windows)

    def shutdown(self):
        self.camera.stop()
        self.stage.close()


def _get_grating_spot(img: np.ndarray, center: Tuple[int, int], window_size: int):
    window = img[center[0] - window_size:center[0] + window_size,
             center[1] - window_size:center[1] + window_size]
    power = np.sum(window)
    return power, window
