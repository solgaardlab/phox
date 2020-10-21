import ctypes

#add types
import numpy as np
from numpy.ctypeslib import ndpointer

lusb = ctypes.CDLL('/usr/local/lib/libusb-1.0.so', mode=ctypes.RTLD_GLOBAL)
xen = ctypes.CDLL('libxeneth.so')

# C Enumerations

# Error codes
I_OK = 0
I_DIRTY = 1
E_BUG = 10000
E_NOINIT = 10001
E_LOGICLOADFAILED = 10002
E_INTERFACE_ERROR = 10003
E_OUT_OF_RANGE = 10004
E_NOT_SUPPORTED = 10005
E_NOT_FOUND = 10006
E_FILTER_DONE = 10007
E_NO_FRAME = 10008
E_SAVE_ERROR = 10009
E_MISMATCHED = 10010
E_BUSY = 10011
E_INVALID_HANDLE = 10012
E_TIMEOUT = 10013
E_FRAMEGRABBER = 10014
E_NO_CONVERSION = 10015
E_FILTER_SKIP_FRAME = 10016
E_WRONG_VERSION = 10017
E_PACKET_ERROR = 10018
E_WRONG_FORMAT = 10019
E_WRONG_SIZE = 10020
E_CAPSTOP = 10021
E_OUT_OF_MEMORY = 10022
E_RFU = 10023

# Used for conversion to string
errcodes = {I_OK: 'I_OK',
            I_DIRTY: 'I_DIRTY',
            E_BUG: 'E_BUG',
            E_NOINIT: 'E_NOINIT',
            E_LOGICLOADFAILED: 'E_LOGICLOADFAILED',
            E_INTERFACE_ERROR: 'E_INTERFACE_ERROR',
            E_OUT_OF_RANGE: 'E_OUT_OF_RANGE',
            E_NOT_SUPPORTED: 'E_NOT_SUPPORTED',
            E_NOT_FOUND: 'E_NOT_FOUND',
            E_FILTER_DONE: 'E_FILTER_DONE',
            E_NO_FRAME: 'E_NO_FRAME',
            E_SAVE_ERROR: 'E_SAVE_ERROR',
            E_MISMATCHED: 'E_MISMATCHED',
            E_BUSY: 'E_BUSY',
            E_INVALID_HANDLE: 'E_INVALID_HANDLE',
            E_TIMEOUT: 'E_TIMEOUT',
            E_FRAMEGRABBER: 'E_FRAMEGRABBER',
            E_NO_CONVERSION: 'E_NO_CONVERSION',
            E_FILTER_SKIP_FRAME: 'E_FILTER_SKIP_FRAME',
            E_WRONG_VERSION: 'E_WRONG_VERSION',
            E_PACKET_ERROR: 'E_PACKET_ERROR',
            E_WRONG_FORMAT: 'E_WRONG_FORMAT',
            E_WRONG_SIZE: 'E_WRONG_SIZE',
            E_CAPSTOP: 'E_CAPSTOP',
            E_OUT_OF_MEMORY: 'E_OUT_OF_MEMORY',
            E_RFU: 'E_RFU'}  # The last one is uncertain

# Frame types, ulong
FT_UNKNOWN = -1
FT_NATIVE = 0
FT_8_BPP_GRAY = 1
FT_16_BPP_GRAY = 2
FT_32_BPP_GRAY = 3
FT_32_BPP_RGBA = 4
FT_32_BPP_RGB = 5
FT_32_BPP_BGRA = 6
FT_32_BPP_BGR = 7

# Pixel size in bytes, used for conversion
pixel_sizes = {FT_UNKNOWN: 0,  # Unknown
               FT_NATIVE: 0,  # Unknown, ask with get_frame_type
               FT_8_BPP_GRAY: 1,
               FT_16_BPP_GRAY: 2,
               FT_32_BPP_GRAY: 4,
               FT_32_BPP_RGBA: 4,
               FT_32_BPP_RGB: 4,
               FT_32_BPP_BGRA: 4,
               FT_32_BPP_BGR: 4}

# GetFrameFlags, ulong
XGF_Blocking = 1
XGF_NoConversion = 2
XGF_FetchPFF = 4
XGF_RFU_1 = 8
XGF_RFU_2 = 16
XGF_RFU_3 = 32

# LoadCalibration flags
# Starts the software correction filter after unpacking the
# calibration data
XLC_StartSoftwareCorrection = 1
XLC_RFU_1 = 2
XLC_RFU_2 = 4
XLC_RFU_3 = 8

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
    def __init__(self, name: str = 'cam://0'):
        self.handle = open_camera(name.encode('UTF-8'), 0, 0)
        self.shape = (get_frame_height(self.handle), get_frame_width(self.handle))

    def start(self):
        return errcodes[start_capture(self.handle)]

    def stop(self):
        return errcodes[stop_capture(self.handle)]

    def frame(self):
        frame = np.zeros(shape=self.shape, dtype=np.int16)
        error = get_frame(self.handle, get_frame_type(self.handle), 1,
                          frame.ctypes.data_as(ndpointer(np.int16)), frame.nbytes)
        if error != 0:
            raise RuntimeError(f'Error: {errcodes[error]}')
        return frame

    def set_integration_time(self, integration_time: int):
        return errcodes[set_property_value(self.handle, 'IntegrationTime'.encode('UTF-8'),
                                           str(integration_time).encode('UTF-8'), 0)]


