import numpy as np


def mzi(theta, phi, n=2, i=0, j=None,
        theta_lower=0, phi_lower=0, epsilon=0, dtype=np.complex128):
    j = i + 1 if j is None else j
    epsilon = epsilon if isinstance(epsilon, tuple) else (epsilon, epsilon)
    cc = np.cos(np.pi / 4 + epsilon[0]) * np.cos(np.pi / 4 + epsilon[1])
    cs = np.cos(np.pi / 4 + epsilon[0]) * np.sin(np.pi / 4 + epsilon[1])
    sc = np.sin(np.pi / 4 + epsilon[0]) * np.cos(np.pi / 4 + epsilon[1])
    ss = np.sin(np.pi / 4 + epsilon[0]) * np.sin(np.pi / 4 + epsilon[1])
    iu, il, eu, el = theta, theta_lower, phi, phi_lower
    mat = np.eye(n, dtype=dtype)
    mzi_mat = np.array([
        [(cc * np.exp(1j * iu) - ss * np.exp(1j * il)) * np.exp(1j * eu),
         1j * (cs * np.exp(1j * iu) + sc * np.exp(1j * il)) * np.exp(1j * el)],
        [1j * (sc * np.exp(1j * iu) + cs * np.exp(1j * il)) * np.exp(1j * eu),
         (cc * np.exp(1j * il) - ss * np.exp(1j * iu)) * np.exp(1j * el)]],
        dtype=dtype)
    mat[i, i], mat[i, j] = mzi_mat[0, 0], mzi_mat[0, 1]
    mat[j, i], mat[j, j] = mzi_mat[1, 0], mzi_mat[1, 1]
    return mat


def balanced_tree(n):
    # this is just defined for powers of 2 for simplicity
    assert np.floor(np.log2(n)) == np.log2(n)
    return [(2 * j * (2 ** k), 2 * j * (2 ** k) + (2 ** k))
            for k in range(int(np.log2(n)))
            for j in reversed(range(n // (2 * 2 ** k)))], n


def diagonal_tree(n, m=0):
    return [(i, i + 1) for i in reversed(range(m, n - 1))], n


def mesh(thetas, phis, network, epsilons=None):
    ts, n = network
    u = np.eye(n)
    epsilons = np.zeros_like(thetas) if epsilons is None else epsilons
    for theta, phi, t, eps in zip(thetas, phis, ts, epsilons):
        u = mzi(theta, phi, n, *t, eps) @ u
    return u


def nullify(vector, i, j=None):
    n = len(vector)
    j = i + 1 if j is None else j
    theta = np.arctan2(np.abs(vector[i]), np.abs(vector[j])) * 2
    phi = np.angle(vector[j]) - np.angle(vector[i])
    mat = mzi(theta, phi, n, i, j)
    nullified_vector = mat @ vector
    return theta, np.mod(phi, 2 * np.pi), nullified_vector


# assumes topologically-ordered tree (e.g. above tree functions)
def analyze(v, tree):
    ts, n = tree
    thetas, phis = np.zeros(len(ts)), np.zeros(len(ts))
    for i, t in enumerate(ts):
        thetas[i], phis[i], v = nullify(v, *t)
    return thetas, phis, v[0]


def reck(u):
    thetas, phis, mzi_lists = [], [], []
    n = u.shape[0]
    for i in range(n - 1):
        tree = diagonal_tree(n, i)
        mzi_lists_i, _ = tree
        thetas_i, phis_i, _ = analyze(u.T[i], tree)
        u = mesh(thetas_i, phis_i, tree) @ u
        thetas.extend(thetas_i)
        phis.extend(phis_i)
        mzi_lists.extend(mzi_lists_i)
    phases = np.angle(np.diag(u))
    return thetas, phis, (mzi_lists, n), phases


def generate(thetas, phis, tree, epsilons=None):
    return mesh(thetas, phis, tree, epsilons)[0]
