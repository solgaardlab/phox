import numpy as np
from .typing import Arraylike


def minmax_scale(img: np.ndarray):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def min_phase_diff(phase):
    return np.minimum(np.abs(phase), np.abs(2 * np.pi + phase), np.abs(-2 * np.pi + phase))