import numpy as np


def dc(epsilon):
    return np.array([
        [np.cos(np.pi / 4 + epsilon), 1j * np.sin(np.pi / 4 + epsilon)],
        [1j * np.sin(np.pi / 4 + epsilon), np.cos(np.pi / 4 + epsilon)]
    ])


def ps(upper, lower):
    return np.array([
        [np.exp(1j * upper), 0],
        [0, np.exp(1j * lower)]
    ])


def mzi(theta, phi, n=2, i=0, j=None,
        theta_upper=0, phi_upper=0, epsilon=0, dtype=np.complex128):
    j = i + 1 if j is None else j
    epsilon = epsilon if isinstance(epsilon, tuple) else (epsilon, epsilon)
    mat = np.eye(n, dtype=dtype)
    mzi_mat = dc(epsilon[1]) @ ps(theta_upper, theta) @ dc(epsilon[0]) @ ps(phi_upper, phi)
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


def mesh(thetas, phis, network, phases=None, epsilons=None):
    ts, n = network
    u = np.eye(n)
    epsilons = np.zeros_like(thetas) if epsilons is None else epsilons
    for theta, phi, t, eps in zip(thetas, phis, ts, epsilons):
        u = mzi(theta, phi, n, *t, eps) @ u
    if phases is not None:
        u = np.diag(phases) @ u
    return u


def nullify(vector, i, j=None):
    n = len(vector)
    j = i + 1 if j is None else j
    theta = -np.arctan2(np.abs(vector[i]), np.abs(vector[j])) * 2
    phi = np.angle(vector[i]) - np.angle(vector[j])
    nullified_vector = mzi(theta, phi, n, i, j) @ vector
    return np.mod(theta, 2 * np.pi), np.mod(phi, 2 * np.pi), nullified_vector


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
    return np.asarray(thetas), np.asarray(phis), (mzi_lists, n), phases


def generate(thetas, phis, tree, epsilons=None):
    return mesh(thetas, phis, tree, epsilons)[0]


def random_complex(n):
    return np.random.randn(n) + np.random.randn(n) * 1j
