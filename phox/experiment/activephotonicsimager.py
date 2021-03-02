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
import panel as pn
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ActivePhotonicsImager:
    def __init__(self, home: Tuple[float, float] = (0, 0), interlayer_distance: float = 0.309,
                 stage_port: str = '/dev/ttyUSB1', laser_port: str = '/dev/ttyUSB0',
                 camera_calibration_filepath: Optional[str] = None, integration_time: int = 20000):
        """

        Args:
            home: Home position for the stage
            interlayer_distance: Distance between layers
            stage_port: Stage serial port str
            laser_port: Laser serial port str
            camera_calibration_filepath: Camera calibration file (for Xenics bobcat camera)
            integration_time: Integration time for the camera
        """
        self.home = home
        self.x_home, self.y_home = home
        self.layer_distance = interlayer_distance
        logger.info('Connecting to camera...')
        self.camera = XCamera(integration_time=integration_time)
        self.integration_time = integration_time
        self.camera.start()
        if camera_calibration_filepath is not None:
            self.camera.load_calibration(camera_calibration_filepath)
        logger.info('Connecting to stage...')
        self.stage = ASI(port=stage_port)
        self.stage.connect()
        logger.info('Connecting to mesh voltage control...')
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
        """

        Args:
            cmap:

        Returns:

        """
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
                   dx: float = 0.001, dy: float = 0.001):
        """

        Args:
            xlim:
            ylim:
            dx:
            dy:

        Returns:

        """
        init_x, init_y = self.stage.where()

        x = pn.widgets.FloatSlider(start=xlim[0], end=xlim[1], step=dx,
                                   value=init_x, name='X Position', format='1[.]000')
        y = pn.widgets.FloatSlider(start=ylim[0], end=ylim[1], step=dy,
                                   value=init_y, name='Y Position', format='1[.]000')
        sync = pn.widgets.Button(name='Sync Stage Position')

        def move_x(*events):
            for event in events:
                if event.name == 'value':
                    self.stage.move(x=event.new)

        def move_y(*events):
            for event in events:
                if event.name == 'value':
                    self.stage.move(y=event.new)

        def sync_(*events):
            new_x, new_y = self.stage.where()
            x.value = new_x
            y.value = new_y

        x.param.watch(move_x, 'value')
        y.param.watch(move_y, 'value')
        sync.on_click(sync_)

        return pn.Column(x, y, sync)

    def wavelength_panel(self, wavelength_range: Tuple[float, float] = (1.530, 1.584), dlam: float = 0.001):
        """
        Panel for dispersion handling

        Args:
            wavelength_range: wavelength range
            dlam: lambda step size for adjustment

        Returns:

        """
        dispersion = pn.widgets.FloatSlider(start=wavelength_range[0], end=wavelength_range[1], step=dlam,
                                            value=self.laser.wavelength, name=r'Wavelength (um)',
                                            format='1[.]000')

        def change_wavelength(*events):
            for event in events:
                if event.name == 'value':
                    self.laser.wavelength = event.new
        dispersion.param.watch(change_wavelength, 'value')

        return dispersion

    def power_panel(self, power_range: Tuple[float, float] = (0.05, 4.25), interval=0.001):
        """

        Args:
            power_range:
            interval:

        Returns:

        """
        power = pn.widgets.FloatSlider(start=power_range[0], end=power_range[1], step=interval,
                                       value=self.laser.power, name='Power (mW)', format='1[.]000')

        def change_power(*events):
            for event in events:
                if event.name == 'value':
                    self.laser.power = event.new

        power.param.watch(change_power, 'value')
        return power

    def current_image(self, n: int = 1, rescale: bool = False):
        """

        Args:
            n:
            rescale:

        Returns:

        """
        imgs = np.asarray([minmax_scale(self.camera.frame(self.camera.integration_time)) if rescale
                           else self.camera.frame(self.camera.integration_time) for _ in range(n)])
        return np.mean(imgs, axis=0)

    def mesh_pics(self, n: int):
        """

        Args:
            n:

        Returns:

        """
        mesh_pics = []
        for m in range(7):
            layer = 1 + m * 3
            self.stage.move(x=self.x_home, y=self.y_home + layer * self.layer_distance)
            time.sleep(3)
            mesh_pics.append(self.current_image(n))
        return mesh_pics

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
        for v in pbar(voltages):
            self.control.write_chan(channel, v)
            time.sleep(wait_time)
            img = self.current_image()
            spots.append([_get_grating_spot(img, center, window_size) for center in centers])
        return np.asarray([s[0] for s in spots]), np.asarray([s[1] for s in spots])

    def centers(self, integration_time: Optional[int] = None, threshold: int = 0) -> List[Tuple[int, int]]:
        """Determine the input and output centers for the MZI (avoids needing to hard-code positions)

        Args:
            integration_time: Integration time (use default from constructor if None)
            threshold: Threshold
            n: Number of images to average to determine count

        Returns:

        """
        self.camera.set_integration_time(self.integration_time if integration_time is None else integration_time)
        img = self.current_image()
        time.sleep(0.2)
        contours = [Polygon(np.fliplr(contour)) for contour in measure.find_contours(img, threshold) if
                    len(contour) > 3]
        contours = [contour for contour in contours if contour.area > 2]
        contour_centers = [(int(contour.centroid.y), int(contour.centroid.x)) for contour in contours]
        return contour_centers

    def center_spot(self, center: Tuple[int, int], n: int, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the center spot power and image

        Args:
            center:
            n:
            window_size:

        Returns:

        """
        img = self.current_image(n)
        return _get_grating_spot(img, center, window_size)

    def calibrate_power(self, center: Tuple[int, int], window_size: int = 10,
                        plim: Tuple[float, float] = (0.05, 4.25), n_steps: int = 421, n_avg: int = 1,
                        pbar: Optional[Callable] = None, wait_time: float = 1,
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
        self.camera.set_integration_time(self.integration_time)
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


def _get_grating_spot(img: np.ndarray, center: Tuple[int, int], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    window = img[center[0] - window_size:center[0] + window_size,
             center[1] - window_size:center[1] + window_size]
    power = np.sum(window)
    return power, window
