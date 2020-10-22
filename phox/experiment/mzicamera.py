from ..instrumentation import ASI, MeshAOControl, XCamera, LaserHP8164A
from typing import Tuple, Callable
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

    def io_centers(self, integration_time: int = 4000, threshold: int = 10000, n: int = 50):
        img = self.current_image(n)
        self.camera.set_integration_time(integration_time)
        time.sleep(0.2)
        contours = [Polygon(np.fliplr(contour)) for contour in measure.find_contours(img, threshold) if
                    len(contour) > 3]
        contours = [contour for contour in contours if contour.area > 2]
        contour_centers = [(int(contour.centroid.y), int(contour.centroid.x)) for contour in contours]
        return contour_centers[0], contour_centers[-1]

    def calibrate_power(self, min_power: float = 0, max_power: float = 4.2, n_steps: int = 421, n_avg: int = 50,
                        window_size: float = 10):
        input_center, _ = self.io_centers()
        laser_powers = np.range(min_power, max_power, n_steps)
        measured_powers = []
        for power in laser_powers:
            self.laser.set_power(power)
            time.sleep(0.5)
            img = self.current_image(n_avg)
            measured_powers.append(_get_grating_spot(img, input_center, window_size)[0])
        return laser_powers, np.asarray(measured_powers)


def _get_grating_spot(img, center, window_size):
    window = img[center[0] - window_size:center[0] + window_size,
                 center[1] - window_size:center[1] + window_size]
    power = np.sum(window)
    return power, window
