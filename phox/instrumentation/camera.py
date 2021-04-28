import ctypes

# add types
from collections import Callable

import numpy as np
import holoviews as hv
from holoviews.streams import Pipe
from numpy.ctypeslib import ndpointer
from threading import Thread, Lock
import logging

from tornado.ioloop import PeriodicCallback

logger = logging.getLogger()
import time
import panel as pn
from tornado import gen

from typing import Optional, List, Tuple

lusb = ctypes.CDLL('/usr/local/lib/libusb-1.0.so', mode=ctypes.RTLD_GLOBAL)
xen = ctypes.CDLL('libxeneth.so')

# C Enumerations

# Used for conversion to string
errcodes = {0: 'I_OK',
            1: 'I_DIRTY',
            10000: 'E_BUG',
            10001: 'E_NOINIT',
            10002: 'E_LOGICLOADFAILED',
            10003: 'E_INTERFACE_ERROR',
            10004: 'E_OUT_OF_RANGE',
            10005: 'E_NOT_SUPPORTED',
            10006: 'E_NOT_FOUND',
            10007: 'E_FILTER_DONE',
            10008: 'E_NO_FRAME',
            10009: 'E_SAVE_ERROR',
            10010: 'E_MISMATCHED',
            10011: 'E_BUSY',
            10012: 'E_INVALID_HANDLE',
            10013: 'E_TIMEOUT',
            10014: 'E_FRAMEGRABBER',
            10015: 'E_NO_CONVERSION',
            10016: 'E_FILTER_SKIP_FRAME',
            10017: 'E_WRONG_VERSION',
            10018: 'E_PACKET_ERROR',
            10019: 'E_WRONG_FORMAT',
            10020: 'E_WRONG_SIZE',
            10021: 'E_CAPSTOP',
            10022: 'E_OUT_OF_MEMORY',
            10023: 'E_RFU'}  # The last one is uncertain

# C functions

# XCHANDLE XC_OpenCamera (const char * pCameraName = "cam://default",
#                         XStatus pCallBack = 0, void * pUser = 0);
open_camera = xen.XC_OpenCamera
open_camera.restype = ctypes.c_uint  # XCHANDLE
#     open_camera.argtypes = (ctypes.c_char_p,)
open_camera.argtypes = (ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)

error_to_string = xen.XC_ErrorToString
error_to_string.restype = ctypes.c_int32
error_to_string.argtypes = (ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32)

is_initialised = xen.XC_IsInitialised
is_initialised.restype = ctypes.c_int32
is_initialised.argtypes = (ctypes.c_int32,)

start_capture = xen.XC_StartCapture
start_capture.restype = ctypes.c_ulong  # ErrCode
start_capture.argtypes = (ctypes.c_int32,)

is_capturing = xen.XC_IsCapturing
is_capturing.restype = ctypes.c_bool
is_capturing.argtypes = (ctypes.c_int32,)

get_frame_size = xen.XC_GetFrameSize
get_frame_size.restype = ctypes.c_ulong
get_frame_size.argtypes = (ctypes.c_int32,)  # Handle

get_frame_type = xen.XC_GetFrameType
get_frame_type.restype = ctypes.c_ulong  # Returns enum
get_frame_type.argtypes = (ctypes.c_int32,)  # Handle

get_frame_width = xen.XC_GetWidth
get_frame_width.restype = ctypes.c_ulong
get_frame_width.argtypes = (ctypes.c_int32,)  # Handle

get_frame_height = xen.XC_GetHeight
get_frame_height.restype = ctypes.c_ulong
get_frame_height.argtypes = (ctypes.c_int32,)  # Handle

get_frame = xen.XC_GetFrame
get_frame.restype = ctypes.c_ulong  # ErrCode
get_frame.argtypes = (ctypes.c_int32, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_uint)

stop_capture = xen.XC_StopCapture
stop_capture.restype = ctypes.c_ulong  # ErrCode
stop_capture.argtypes = (ctypes.c_int32,)

close_camera = xen.XC_CloseCamera
# Returns void
close_camera.argtypes = (ctypes.c_int32,)  # Handle

# Calibration
load_calibration = xen.XC_LoadCalibration
load_calibration.restype = ctypes.c_ulong  # ErrCode
# load_calibration.argtypes = (ctypes.c_int32, ctypes.c_char_p, ctypes.c_ulong)

# ColourProfile
load_colour_profile = xen.XC_LoadColourProfile
load_colour_profile.restype = ctypes.c_ulong
load_colour_profile.argtypes = (ctypes.c_char_p,)

# Settings
load_settings = xen.XC_LoadSettings
load_settings.restype = ctypes.c_ulong
load_settings.argtypes = (ctypes.c_char_p, ctypes.c_ulong)

# FileAccessCorrectionFile
set_property_value = xen.XC_SetPropertyValue
set_property_value.restype = ctypes.c_ulong  # ErrCode
# set_property_value.argtypes = (ctypes.c_int32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p)

# Set property value
set_property_value_f = xen.XC_SetPropertyValueF
set_property_value_f.restype = ctypes.c_ulong  # ErrCode
# set_property_value.argtypes = (ctypes.c_int32, ctypes.c_char_p, ctypes.c_double, ctypes.c_char_p)


class XCamera:
    def __init__(self, name: str = 'cam://0', integration_time: int = 5000,
                 spots: Optional[List[Tuple[int, int, int, int]]] = None):
        """Xenics Camera (XCamera)

        Args:
            name: camera URL
            integration_time: Default integration time for the camera
            spots: list of spots to track during frame loop
        """
        self.handle = open_camera(name.encode('UTF-8'), 0, 0)
        self.shape = (get_frame_height(self.handle), get_frame_width(self.handle))
        self._current_frame = None
        self.calibrate = False
        self.thread = None
        self.started_frame_loop = False
        self.started = False
        self.frame_lock = Lock()
        self.background_reference = None
        self.set_integration_time(integration_time)
        self.integration_time = integration_time * 1e-6
        self.livestream_pipe = Pipe(data=[])
        self.spots = [] if spots is None else spots
        self.spot_powers = []
        self.livestream_on = False

    def start(self) -> int:
        self.started = True
        return errcodes[start_capture(self.handle)]

    def stop(self) -> int:
        self.started = False
        return errcodes[stop_capture(self.handle)]

    def start_frame_loop(self, on_frame: Optional[Callable] = None):
        if self.started_frame_loop:
            logger.warning('Cannot start a frame loop that has already been started. '
                           'Use end_frame_loop to stop current frame loop.')
            return
        self.started_frame_loop = True
        self.thread = Thread(target=self._frame_loop, args=(on_frame,))
        self.thread.start()

    def _frame_loop(self, on_frame: Optional[Callable] = None):
        while self.started_frame_loop:
            frame = self._frame()
            if self.spots is not None:
                self.spot_powers = np.asarray([_get_grating_spot(frame, (s[0], s[1]), (s[2], s[3]))[0]
                                               for s in self.spots])
            if on_frame is not None:
                on_frame(frame)
            with self.frame_lock:
                self._current_frame = frame.copy()

    def _livestream_loop(self, on_frame: Optional[Callable] = None):
        while self.started_frame_loop:
            frame = self._frame()
            if self.spots is not None:
                self.spot_powers = np.asarray([_get_grating_spot(frame, (s[0], s[1]), (s[2], s[3]))[0]
                                               for s in self.spots])
            if on_frame is not None:
                on_frame(frame)
            with self.frame_lock:
                self._current_frame = frame.copy()

    def stop_frame_loop(self):
        self.started_frame_loop = False
        self.thread.join()

    def frame(self, wait_time: float = 0) -> np.ndarray:
        if self.started_frame_loop:
            time.sleep(wait_time)
            with self.frame_lock:
                frame = self._current_frame.copy()
            return frame
        else:
            return self._frame()

    def _frame(self) -> np.ndarray:
        if not self.started:
            raise RuntimeError('Camera must be started to capture a frame.')
        frame = np.zeros(shape=self.shape, dtype=np.uint16)
        error = get_frame(self.handle, get_frame_type(self.handle), 1,
                          frame.ctypes.data_as(ndpointer(np.uint16)), frame.nbytes)
        if error != 0:
            raise RuntimeError(f'Camera Error: {errcodes[error]}')

        return frame if self.background_reference is None else frame.astype(np.float) - self.background_reference.astype(np.float)

    def set_integration_time(self, integration_time: int):
        return errcodes[set_property_value(self.handle, 'IntegrationTime'.encode('UTF-8'),
                                           str(integration_time).encode('UTF-8'), 0)]

    def load_calibration(self, filepath: str):
        self.calibrate = True
        return errcodes[load_calibration(self.handle, filepath.encode('UTF-8'), 1)]

    def __exit__(self):
        if self.started_frame_loop:
            self.stop_frame_loop()
        self.stop()

    def livestream_panel(self, cmap='hot'):
        """

        Args:
            cmap: colormap for the livestream

        Returns:
            A video livestream for the camera

        """
        bounded_img = lambda data: hv.Image(data, bounds=(0, 0, 640, 512))
        dmap = hv.DynamicMap(bounded_img, streams=[self.livestream_pipe]).opts(
            width=640, height=512, show_grid=True, colorbar=True, cmap=cmap,
            shared_axes=False).redim.range(z=(0, 2 ** 15))
        livestream_toggle = pn.widgets.Toggle(name='Livestream', value=False)
        self.livestream_pipe.send(np.fliplr(self.frame().astype(np.float)))

        @gen.coroutine
        def update_plot():
            self.livestream_pipe.send(np.fliplr(self.frame().astype(np.float)))

        cb = PeriodicCallback(update_plot, 100)

        def change_livestream(*events):
            for event in events:
                if event.name == 'value':
                    self.livestream_on = bool(event.new)
                    if self.livestream_on:
                        cb.start()
                    else:
                        cb.stop()

        livestream_toggle.param.watch(change_livestream, 'value')

        return pn.Column(dmap, livestream_toggle)


def _get_grating_spot(img: np.ndarray, center: Tuple[int, int], window_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    window = img[center[0] - window_size[0]:center[0] + window_size[0],
             center[1] - window_size[1]:center[1] + window_size[1]]
    power = np.sum(window)
    return power, window
