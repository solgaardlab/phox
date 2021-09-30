import numpy as np
from .typing import Arraylike
from neurophox.components import SMMZI
import hashlib


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


def psk_hash(input_str):
    hasher1 = hashlib.sha3_256()
    hasher1.update(input_str)

    x = []
    for k in range(64):
        x.append(0)
    for i in range(0, 64, 2):
        j = i >> 1
        x[i] = hasher1.digest()[j] >> 4
        x[i + 1] = hasher1.digest()[j] & 0x0F

    return x, np.asarray(x) / 16 * 2 * np.pi


def pow_hash_matmul(M, input):
    hasher1 = hashlib.sha3_256()
    hasher1.update(input)

    x = []
    for k in range(64):
        x.append(0)
    for i in range(0, 64, 2):
        j = i >> 1
        x[i] = hasher1.digest()[j] >> 4
        x[i + 1] = hasher1.digest()[j] & 0x0F

    y = []
    for i in range(len(x)):
        y.append(0)
        for j in range(len(x)):
            y[i] += M[i][j] * x[j]
        y[i] = y[i] >> 10
    return x, y
