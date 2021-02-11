import numpy as np
import cv2


def minmax_scale(img: np.ndarray):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def xframe_to_jpg(xframe: np.ndarray):
    # xframe is a xenics camera frame in uint16 format
    ret, jpeg = cv2.imencode('.jpg', (xframe // 256).astype(np.uint8))
    return jpeg.tobytes()
