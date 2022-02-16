import hashlib
import numpy as np
import time


def psk_hash(input_str):
    hasher1 = hashlib.sha3_256()
    hasher1.update(input_str)

    x = np.zeros(64)
    for i in range(0, 64, 2):
        j = i >> 1
        x[i] = hasher1.digest()[j] >> 4
        x[i + 1] = hasher1.digest()[j] & 0x0F

    return x, np.array(x) / 16 * 2 * np.pi


def pow_hash_matmul(M, input):
    hasher1 = hashlib.sha3_256()
    hasher1.update(input)

    x = np.zeros(64)
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


def random_lh_matrix(n_vecs, n, num_vals, normed=False, is_complex=False):
    offset = num_vals - 1
    v = (2 * np.random.randint(num_vals, size=(n_vecs, n)) - offset) + ((2 * np.random.randint(num_vals, size=(n_vecs, n)) - offset) * 1j) * is_complex
    return v / np.sqrt(np.sum(v ** 2, axis=1)[:, np.newaxis]) if normed else v


def svd_demo(chip, u, d, v, x, p=0, wait_time=0.05):
    if p > 0:
        v = np.roll(v, p)
        u = np.roll(u, p, axis=1)
        d = np.roll(d, p)
    chip.set_unitary(v)
    chip.set_input(np.array((*x, 0)))
    time.sleep(wait_time)
    res = np.sqrt(np.maximum(chip.fractional_right[:4], 0)) * np.exp(1j * np.angle(v @ x)) * np.linalg.norm(x)
    res = res * d
    chip.set_unitary(u)
    chip.set_input(np.array((*res, 0)))
    time.sleep(wait_time)
    return np.linalg.norm(res) * np.sqrt(np.maximum(chip.fractional_right[:4], 0))
