import hashlib
import numpy as np

from dataclasses import dataclass
from scipy.linalg import block_diag
import random


def data_to_sha256_vec(data):
    hasher1 = hashlib.sha3_256()
    hasher1.update(data.encode('ascii'))
    b = np.frombuffer(np.binary_repr(int(hasher1.hexdigest(), 16)).encode('utf-8'), dtype='S1').astype(int)
    return np.hstack((np.zeros(256 - b.size), b))


def sha256_str(val: str):
    val = val.encode('utf-8') if isinstance(val, str) else val
    return np.binary_repr(int(hashlib.sha256(val).hexdigest(), 16)).encode('utf-8')


def double_sha256_str(val: str):
    return sha256_str(sha256_str(val))


@dataclass
class LHNode:
    left: "LHNode"
    right: "LHNode"
    value: bytes


@dataclass
class LHMerkleTree:
    """For computational efficiency, stores large amount of transaction data in Merkle trees.

    Attributes:
        values: The list of values to store in the Merkle tree (forming the leaves of Merkle tree).

    """
    values: list[str]

    def __post_init__(self) -> None:
        leaves: list[LHNode] = [LHNode(None, None, double_sha256_str(e)) for e in self.values]
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1:][0])  # duplicate last elem if odd number of elements
        self.root: LHNode = self._build_tree(leaves)

    def _build_tree(self, nodes: list[LHNode]) -> LHNode:
        if len(nodes) == 2:
            return LHNode(nodes[0], nodes[1], double_sha256_str(nodes[0].value + nodes[1].value))
        left = self._build_tree(nodes[:len(nodes) // 2])
        right = self._build_tree(nodes[len(nodes) // 2:])
        value: bytes = double_sha256_str(left.value + right.value)
        return LHNode(left, right, value)

    @property
    def root_hash(self) -> bytes:
        return self.root.value


@dataclass
class LHBlock:
    """The LightHash block

    Attributes:
        data: The data organized using a LightHash Merkle Tree
        hash: The hash of this block (updated in init method upon solving the puzzle)
        last_block: The last block in this blockchain
        difficulty: The difficulty of the problem/puzzle based on number of zero bits at beginning.
        nonce: The varying number in the block that is used as a tuning knob to solve the puzzle
        n: number of inputs and outputs for the optical device
        k: the numerical resolution, or number of integers spaced two apart symmetrically about 0.

    """
    data: LHMerkleTree
    hash: str = None  # this will be overwritten during block creation
    last_block: "LHBlock" = None  # genesis block if None
    difficulty: int = 6  # this is much larger in practice
    nonce: int = 0
    n: int = 64
    k: int = 2

    def __post_init__(self):
        n, k = self.n, self.k
        self.last_hash = self.last_block.hash if self.last_block is not None else (np.binary_repr(0) * 256)
        np.random.seed(int(self.last_hash) % 32)  # use the previous hash to seed the randomness of Q

        # generate Q matrix and find all necessary constants via random trials
        q = block_diag(*[random_lh_matrix(n, n, k).squeeze() for _ in range(256 // n)])
        x = q[:n, :n] @ random_lh_input(n, 10000, False)
        hist, bins = np.histogram(np.abs(x.flatten()), bins=int(np.max(x)), density=True)
        cdf = np.cumsum(hist)
        thresh = bins[np.argmin(np.abs(cdf - 0.5))]
        idx = int(thresh)
        self.yth = (bins[idx] + bins[idx + 2]) / 2
        self.q = q

        solved = False
        while not solved:
            self.nonce = random.getrandbits(256)
            sol = lighthash(q, self.lighthash_input, self.yth)
            solved = sol < 2 ** (256 - self.difficulty)
        self.hash = np.binary_repr(sol)

    @property
    def lighthash_input(self):
        return self.last_hash + self.data.root_hash.decode('utf-8') + np.binary_repr(self.nonce)


def lighthash(q, data, yth):
    """The core LightHash function is simulated on a digital platform here.

    Args:
        q: LightHash matrix
        data: Data to solve lighthash
        yth: Threshold

    Returns:
        The lighthash output bits.

    """
    b_in = data_to_sha256_vec(data)
    x = 2 * b_in - 1
    y = q @ x
    b = (y > yth)
    b_out = np.logical_xor(b_in, b).astype(int)
    return b_out.dot(2 ** np.arange(b_out.size, dtype=object)[::-1])


def generate_random_lh_transactions(people, limit: int=100, num_tx: int =100):
    """Generate random transactions to tes the LightHash emulator.

    Args:
        people: The people in the universe for these transactions
        limit: Limit of the number of coins that can be sent to people
        num_tx: Number of total transactions in this segment

    Returns:

    """
    transaction_data = ""
    for i in range(num_tx):
        person_a, person_b = np.random.choice(people, 2, replace=False)
        amount = np.random.choice(limit)
        transaction_data += f"{person_a} sent {person_b} {amount} coin.\n"
    return transaction_data


def random_lh_matrix(n_vecs, n, num_vals, normed=False):
    offset = num_vals - 1
    v = (2 * np.random.randint(num_vals, size=(n_vecs, n)) - offset)
    return v / np.sqrt(np.sum(v ** 2, axis=1)[:, np.newaxis]) if normed else v


def random_lh_input(n, n_trials, normed=True):
    v = (2 * np.random.randint(2, size=(n, n_trials)) - 1)
    return v / np.sqrt(n) if normed else v
