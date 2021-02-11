from ..instrumentation import ASI, MeshAOControl, XCamera, LaserHP8164A
from ..utils import minmax_scale
from typing import Tuple, Callable, Optional, List
import numpy as np
import time

from skimage import measure
from shapely.geometry import Polygon

import holoviews as hv
from holoviews import opts
from holoviews.streams import Pipe
# from bokeh.models.formatters import PrintfTickFormatter

import panel as pn
from panel.interact import interact

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class MZICamera:
    def __init__(self, home: Tuple[float, float] = (0, 0), layer_distance: float = 0.309,
                 stage_port: str = '/dev/ttyUSB0', laser_port: str = '/dev/ttyUSB1',
                 camera_calibration_filepath: Optional[str] = None):
        self.home = home
        self.x_home, self.y_home = home
        self.layer_distance = layer_distance
        logger.info('Connecting to camera...')
        self.camera = XCamera()
        self.camera.start()
        if camera_calibration_filepath is not None:
            self.camera.load_calibration(camera_calibration_filepath)
        logger.info('Connecting to stage...')
        self.stage = ASI(port=stage_port)
        self.stage.connect()
        logger.info('Connecting to mesh voltagecontrol...')
        self.control = MeshAOControl()
        logger.info('Connecting to laser...')
        self.laser = LaserHP8164A(port=laser_port)
        self.laser.connect()
        self.livestream_pipe = Pipe(data=[])
        logger.info('Turning laser off...')
        self.laser.state = 0
        time.sleep(1)
        logger.info('Taking camera reference...')
        self.camera.background_reference = self.camera.frame()
        time.sleep(1)
        logger.info('Turning laser back on...')
        self.laser.state = 1

    def go_home(self):
        self.stage.move(*self.home)

    def livestream_panel(self, cmap='hot'):
        dmap = hv.DynamicMap(hv.Image, streams=[self.livestream_pipe]).opts(
            width=640, height=512, show_grid=True, colorbar=True, xaxis=None, yaxis=None, cmap=cmap)

        def update_plot(img):
            time.sleep(0.1)
            self.livestream_pipe.send(img.astype(np.float))
        scalebar = hv.Text(0.4, 0.4, '50 um').opts(
            text_align='center', text_baseline='middle',
            text_color='green', text_font='Arial') * hv.Path([(0.35, 0.45), (0.45, 0.45)]).opts(color='green',
                                                                                                line_width=4)
        self.camera.start_frame_loop(update_plot)
        return dmap.opts(opts.Image(axiswise=True, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))) * scalebar

    def move_panel(self, xlim: Tuple[float, float] = (-1, 1), ylim: Tuple[float, float] = (-3.09, 3.09),
              dx: float = 0.01, dy: float = None):
        init_x, init_y = self.stage.where()
        dy = dy if dy is not None else self.layer_distance

        @interact(x=(*xlim, dx), y=(*ylim, dy))
        def move(x=init_x, y=init_y):
            self.stage.move(x, y)
        return move

    def wavelength_panel(self, wavelength_range: Tuple[float, float] = (1.530, 1.584), dlam: float = 0.001):
        """
        Panel for dispersion handling

        Args:
            wavelength_range: wavelength range
            dlam: lambda step size for adjustment

        Returns:

        """
        dispersion = pn.widgets.FloatSlider(start=wavelength_range[0], end=wavelength_range[1], step=dlam,
                                            value=self.laser.wavelength, name='Wavelength', format='1[.]000')

        def change_wavelength(*events):
            for event in events:
                if event.name == 'value':
                    self.laser.wavelength = event.new
        dispersion.param.watch(change_wavelength, 'value')
        return dispersion

    def power_panel(self, power_range: Tuple[float, float] = (0.05, 4.25), interval=0.01):
        @interact(power=(*power_range, interval))
        def laser_power(power=self.laser.power):
            self.laser.power = power
        return laser_power

    def current_image(self, n: int = 1, rescale: bool = False):
        imgs = np.asarray([minmax_scale(self.camera.frame()) if rescale
                           else self.camera.frame() for _ in range(n)])
        return np.mean(imgs, axis=0)

    def mesh_pics(self, n: int):
        mesh_pics = []
        for m in range(7):
            layer = 1 + m * 3
            self.stage.move(x=self.x_home, y=self.y_home + layer * self.layer_distance)
            time.sleep(3)
            mesh_pics.append(self.current_image(n))
        return mesh_pics

    def mzi_sweep(self, voltages: np.ndarray, channel: int, spot_extract_integration_time: int, threshold: int,
                  ps_sweep_integration_time: int, pbar: Callable, n: int, window_size: int):
        output_powers = []
        input_powers = []
        input_center, output_center = self.centers(spot_extract_integration_time, threshold, n)
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

    def centers(self, integration_time: int = 4000, threshold: int = 0, n: int = 50):
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
                        plim: Tuple[float, float] = (0.05, 4.25), n_steps: int = 421, n_avg: int = 1,
                        integration_time: int = 4000, pbar: Optional[Callable] = None, wait_time: float = 1,
                        init_wait_time: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calibrate laser power to the spot measurements in a given grating spot
           (should be done for each grating spot!)

        Args:
            center: (x, y) for the center of the calibration image
            window_size: window size in the calibration image to compute the power
            plim: power range (pmin, pmax) for the calibration
            n_steps: number of steps in the sweep (resolution)
            n_avg: number of times to average to determine the power
            integration_time: integration time for the laser power calibration
            pbar: callable for progressbar (for longer calibrations)

        Returns:
            A tuple of laser powers and measured powers associated with the calibration

        """
        laser_powers = np.linspace(*plim, n_steps)
        measured_powers = []
        measured_windows = []
        self.camera.set_integration_time(integration_time)
        iterator = pbar(laser_powers) if pbar is not None else laser_powers
        self.laser.power = plim[0]
        time.sleep(init_wait_time)
        for power in iterator:
            self.laser.power = power
            time.sleep(wait_time)
            img = self.current_image(n_avg)
            power, window = _get_grating_spot(img, center, window_size)
            measured_powers.append(power)
            measured_windows.append(window)
        return laser_powers, np.asarray(measured_powers), np.stack(measured_windows)

    def dispersion(self, centers: List[Tuple[int, int]], window_size: int = 20,
                   wlim: Tuple[float, float] = (1.53, 1.57), n_steps: int = 401, n_avg: int = 1,
                   integration_time: int = 4000, pbar: Optional[Callable] = None, wait_time: float = 1,
                   init_wait_time: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Determine the dispersion relationship

        Args:
            centers: list of (x, y) for the center of the dispersion images
            window_size: window size for spots in the image to compute the power
            plim: power range (pmin, pmax) for the calibration
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
            img = self.current_image(n_avg)
            res = [_get_grating_spot(img, center, window_size) for center in centers]
            power, window = [r[0] for r in res], [r[1] for r in res]
            measured_powers.append(np.asarray(power))
            measured_windows.append(np.asarray(window))
        return wavelengths, np.asarray(measured_powers), np.asarray(measured_windows)

    def shutdown(self):
        self.camera.stop()
        self.stage.close()


def _get_grating_spot(img: np.ndarray, center: Tuple[int, int], window_size: int):
    window = img[center[0] - window_size:center[0] + window_size,
             center[1] - window_size:center[1] + window_size]
    power = np.sum(window)
    return power, window
