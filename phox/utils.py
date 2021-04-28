import numpy as np
from .typing import Arraylike
from neurophox.components import SMMZI
from typing import Tuple, Union


def minmax_scale(img: np.ndarray):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def vector_to_phases(vector: Arraylike, lower_theta: Arraylike, lower_phi: Arraylike):
    """Use neurophox to convert vector to phases

    Args:
        vector: Vector to program
        lower_theta: List of booleans signifying whether theta is on lower internal arm of MZI
        lower_phi: List of booleans signifying whether phi is on lower external arm of MZI

    Returns:
        :math:`\\theta` and :math:`\\phi` phases for normalized vector

    """
    n = len(vector)
    thetas, phis = [], []
    v = vector
    for i in range(n - 1):
        j = n - 2 - i
        v, _, theta, phi = SMMZI.nullify(v, j, lower_theta=lower_theta[j], lower_phi=lower_phi[j])
        thetas.append(theta)
        phis.append(phi)
    return thetas[::-1], phis[::-1]


def phases_to_vector(thetas: Arraylike, phis: Arraylike, lower_theta: Arraylike, lower_phi: Arraylike):
    """Use neurophox to convert :math:`\\theta` phases to vector

    Args:
        thetas: :math:`\\theta` phases
        phis: :math:`\\phi` phases
        lower_theta: List of booleans signifying whether theta is on lower internal arm of MZI
        lower_phi: List of booleans signifying whether phi is on lower external arm of MZI

    Returns:
        Vector from :math:`\\theta` and :math:`\\phi` phases

    """
    n = len(thetas)
    v = np.zeros(n + 1)
    v[0] = 1
    for i in range(n):
        theta, phi = thetas[i], phis[i]
        v = v @ SMMZI(theta, phi, hadamard=False, lower_theta=lower_theta[i],
                      lower_phi=lower_phi[i]).givens_rotation(n + 1, i)
    return v.conj()


def cal_v_power(v, a, b, p0, p1, p2, p3):
    # take the absolute value of a to ensure a is positive!
    return np.abs(a) * np.sin((p0 * v ** 3 + p1 * v ** 2 + p2 * v + p3)) ** 2 + b


def cal_phase_v(x, q0, q1, q2, q3):
    return q0 * x ** 3 + q1 * x ** 2 + q2 * x + q3
