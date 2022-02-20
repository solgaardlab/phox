from .viz import Dataset, add_bias, light_rdbu, dark_bwr
from ..experiment.amf420mesh import AMF420Mesh
import jax
import jax.numpy as jnp
import optax
import haiku as hk

from simphox.circuit import triangular

import numpy as np
from tensorflow.keras import Model
from scipy.special import softmax

import time
import pickle


PHI_LOCS = [(0, 0), (2, 1), (4, 2), (4, 0), (6, 1), (8, 0)]
THETA_LOCS = [(1, 0), (3, 1), (5, 2), (5, 0), (7, 1), (9, 0)]


def normalize(x):
    return x / np.linalg.norm(x)


def extract_gradients_from_fields(forward, backward):
    gradients = -(backward * forward).imag
    return {
        'theta': [gradients[tl[0], tl[1]] for tl in THETA_LOCS],
        'phi': [gradients[pl[0], pl[1]] for pl in PHI_LOCS]
    }


def extract_gradients_from_powers(forward_p, backward_p, sum_p):
    gradients = (sum_p - forward_p - backward_p) / 2
    return {
        'theta': np.array([gradients[tl[0], :, tl[1]] for tl in THETA_LOCS]).T,
        'phi': np.array([gradients[pl[0], :, pl[1]] for pl in PHI_LOCS]).T
    }


class Tri(hk.Module):
    def __init__(self, n, activation=None, name=None):
        super().__init__(name=name)
        self.output_size = self.n = n
        self._network = triangular(n)
        self._network.gammas = np.zeros_like(self._network.gammas) # useful hack for this demo
        self.matrix = self._network.matrix_fn(use_jax=True)
        self.activation = activation if activation is not None else (lambda x: x)

    def __call__(self, x):
        theta = hk.get_parameter("theta",
                                 shape=self._network.thetas.shape,
                                 init=hk.initializers.Constant(self._network.thetas))
        phi = hk.get_parameter("phi",
                               shape=self._network.phis.shape,
                               init=hk.initializers.Constant(self._network.phis))
        gamma = hk.get_parameter("gamma",
                                 shape=self._network.gammas.shape,
                                 init=hk.initializers.Constant(self._network.gammas))
        return self.activation((self.matrix((theta, phi, gamma)) @ x.T).T)


def optical_softmax(x):
    y = jnp.abs(x) ** 2
    return jnp.vstack((jnp.sum(y[:, :2], axis=1),
                       jnp.sum(y[:, 2:], axis=1))).T


def softmax_cross_entropy(logits, labels):
    return -jnp.sum(jax.nn.log_softmax(logits) * labels, axis=-1)


def optical_softmax_cross_entropy(labels):
    return lambda x: jnp.mean(softmax_cross_entropy(optical_softmax(x), labels)).astype(jnp.float64)


def onn(x, y):
    logits = hk.Sequential([
        Tri(4, activation=jnp.abs, name='layer1'),
        Tri(4, activation=jnp.abs, name='layer2'),
        Tri(4, activation=jnp.abs, name='layer3')
    ])(x)
    return optical_softmax_cross_entropy(y)(logits)


onn_t = hk.without_apply_rng(hk.transform(onn))


@jax.jit
def loss(params, X, y):
    return onn_t.apply(params, X, y)


def get_update_fn(opt):
    @jax.jit
    def update(params: hk.Params, opt_state: optax.OptState, X, y):
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(loss)(params, X, y)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    return update


def get_gradient_predictions(onn_layers, X, y, idx=0):
    pred = dict()
    pred['input_1'] = X[idx]
    for layer in (1, 2, 3):
        pred[f'forward_{layer}'] = onn_layers[f'layer{layer}'].propagate(pred[f'input_{layer}'])
        pred[f'input_{layer + 1}'] = np.abs(pred[f'forward_{layer}'][-1]) + 0j
    cost_fn = optical_softmax_cross_entropy(y[idx])
    pred[f'adjoint_4'] = np.array(jax.grad(cost_fn)(jnp.array(pred[f'input_4'])[np.newaxis, :])).squeeze()
    for layer in (3, 2, 1):
        pred[f'error_{layer}'] = pred[f'adjoint_{layer + 1}'].real * np.exp(
            -1j * np.angle(pred[f'forward_{layer}'][-1]))  # abs value nonlinearity
        pred[f'backward_{layer}'] = onn_layers[f'layer{layer}'].propagate(pred[f'error_{layer}'], back=True)[::-1]
        pred[f'adjoint_{layer}'] = pred[f'backward_{layer}'][0]
    for layer in (1, 2, 3):
        pred[f'sum_{layer}'] = onn_layers[f'layer{layer}'].propagate(
            pred[f'input_{layer}'] - 1j * pred[f'adjoint_{layer}'].conj())

    pred['gradients'] = {
        f'layer{layer}': extract_gradients_from_fields(pred[f'forward_{layer}'][::2],
                                                       pred[f'backward_{layer}'][::2])
        for layer in (1, 2, 3)
    }
    return pred


class BackpropAccuracyTest:
    """This class tests a specific 3-layer photonic neural network.

    """
    def __init__(self, chip: AMF420Mesh, params_list, X, y, idx_list, iteration, wait_time: float = 0.05):
        self.X = X
        self.y = y
        self.chip = chip
        self.onn_layers = {}
        self.meas = {}
        self.idx_list = idx_list
        self.wait_time = wait_time
        self.onn_layers = {'layer1': triangular(4),
                           'layer2': triangular(4),
                           'layer3': triangular(4)}
        for layer in (1, 2, 3):
            self.onn_layers[f'layer{layer}'].params = (
                np.mod(params_list[iteration][f'layer{layer}']['theta'], 2 * np.pi),
                np.mod(params_list[iteration][f'layer{layer}']['phi'], 2 * np.pi),
                np.mod(params_list[iteration][f'layer{layer}']['gamma'], 2 * np.pi)
            )
        self.pred = get_gradient_predictions(self.onn_layers, X, y, idx_list[0])
        pred_list = [get_gradient_predictions(self.onn_layers, X, y, i) for i in idx_list]
        self.pred = {key: np.array([p[key] for p in pred_list]) for key in self.pred}
        for key in self.pred:
            if self.pred[key].ndim == 3:
                self.pred[key] = self.pred[key].transpose((1, 0, 2))

    def _sum_measure(self):
        for layer in (1, 2, 3):
            sum_input = self.meas[f'input_{layer}'] - 1j * self.meas[f'adjoint_{layer}'].conj()
            self.chip.set_unitary_phases(self.onn_layers[f'layer{layer}'].thetas, self.onn_layers[f'layer{layer}'].phis)
            self.meas[f'sum_{layer}'] = self.chip.matrix_prop(sum_input)

    def _forward_measure(self, phase_cheat: bool = False):
        self.chip.set_transparent()
        self.meas['input_1'] = np.array([self.X[idx] for idx in self.idx_list])
        for layer in (1, 2, 3):
            self.chip.set_unitary_phases(self.onn_layers[f'layer{layer}'].thetas, self.onn_layers[f'layer{layer}'].phis)
            self.meas[f'forward_{layer}'] = self.chip.matrix_prop(self.meas[f'input_{layer}'])
            self.meas[f'input_{layer + 1}'] = np.sqrt(np.abs(self.meas[f'forward_{layer}'][-1][:, :4]))
            if not phase_cheat:
                self.meas[f'forward_out_{layer}'] = self.chip.coherent_batch(self.meas[f'input_{layer}'])
            else:
                self.meas[f'forward_out_{layer}'] = self.pred[f'forward_{layer}'][-1]
            self.chip.set_output_transparent()

    def _backward_measure(self, phase_cheat: bool = False):
        cost_fn = [optical_softmax_cross_entropy(self.y[idx]) for idx in self.idx_list]
        self.meas['adjoint_4'] = np.array([jax.grad(cost_fn[i])(self.meas[f'input_4'][i:i + 1, :]).squeeze()
                                           for i in range(len(self.idx_list))])
        self.chip.set_transparent()
        self.chip.toggle_propagation_direction()
        for layer in (3, 2, 1):
            forward_phasors = np.exp(-1j * np.angle(self.meas[f'forward_out_{layer}']))
            self.meas[f'error_{layer}'] = self.meas[f'adjoint_{layer + 1}'].real * forward_phasors
            self.chip.set_unitary_phases(self.onn_layers[f'layer{layer}'].thetas, self.onn_layers[f'layer{layer}'].phis)
            self.meas[f'backward_{layer}'] = self.chip.matrix_prop(self.meas[f'error_{layer}'])
            if phase_cheat:
                self.meas[f'backward_out_{layer}'] = self.pred[f'backward_{layer}'][0]
            else:
                self.meas[f'backward_out_{layer}'] = self.chip.coherent_batch(self.meas[f'error_{layer}'])
            backward_phasors = np.exp(1j * np.angle(self.meas[f'backward_out_{layer}']))
            self.meas[f'adjoint_{layer}'] = np.sqrt(np.abs(self.meas[f'backward_{layer}'][0][:, :4])) * backward_phasors
            self.chip.set_output_transparent()
        self.chip.toggle_propagation_direction()

    def run(self, phase_cheat: bool = False):
        self.chip.reset_control()
        self._forward_measure(phase_cheat)
        self._backward_measure(phase_cheat)
        self._sum_measure()
        self.chip.reset_control()
        self.meas['gradients'] = {f'layer{layer}': extract_gradients_from_powers(
            self.meas[f'forward_{layer}'][:11, :, :4],
            self.meas[f'backward_{layer}'][:11, :, :4],
            self.meas[f'sum_{layer}'][:11, :, :4]
        ) for layer in (1, 2, 3)}



#TODO(sunil): fix a lot of boilerplate code.
class ONN2D:
    def __init__(self, dataset: Dataset, dataset_test: Dataset,
                 n_layers: int,
                 y_sim: np.ndarray = None, y_sim_test: np.ndarray = None,
                 y_onn: np.ndarray = None, y_onn_test: np.ndarray = None):
        self.n_layers = n_layers
        self.unitaries = []
        self.dataset = dataset
        self.dataset_test = dataset_test
        self.y_sim = [] if y_sim is None else np.asarray(y_sim)
        self.y_sim_test = [] if y_sim_test is None else np.asarray(y_sim_test)
        self.y_onn = [] if y_onn is None else y_onn
        self.y_onn_test = [] if y_onn_test is None else y_onn_test

    def set_model(self, model: Model):
        self.unitaries = [model.layers[i].matrix.conj().T for i in range(self.n_layers)]

    def onn(self, input_vector: np.ndarray, mesh: AMF420Mesh, meas_delay: float = 0.2, factor: float = 3):
        outputs = input_vector
        for u in self.unitaries:
            mesh.set_input(outputs)
            mesh.set_unitary(u)
            time.sleep(meas_delay)
            outputs = np.sqrt(np.abs(mesh.fractional_right[:4]))
        return softmax(factor ** 2 * np.asarray((np.sum(outputs[:2] ** 2), np.sum(outputs[2:] ** 2))))

    def classify_train(self, mesh, model: Model, pbar=None, meas_time: float = 0.2):
        self.y_sim = []
        self.y_onn = []
        iterator = pbar(range(len(self.dataset.X))) if pbar else range(len(self.dataset.X))
        for i in iterator:
            self.y_sim.append(model(self.dataset.X[i]))
            self.y_onn.append(self.onn(self.dataset.X[i], mesh, meas_time))
        self.y_sim = np.asarray(self.y_sim)[:, 0, :]
        self.y_onn = np.asarray(self.y_onn)

    def classify_test(self, mesh, model: Model, pbar=None, meas_time: float = 0.2):
        self.y_sim_test = []
        self.y_onn_test = []
        iterator = pbar(range(len(self.dataset_test.X))) if pbar else range(len(self.dataset_test.X))
        for i in iterator:
            self.y_sim_test.append(model(self.dataset_test.X[i]))
            self.y_onn_test.append(self.onn(self.dataset_test.X[i], mesh, meas_time))
        self.y_sim_test = np.asarray(self.y_sim_test)[:, 0, :]
        self.y_onn_test = np.asarray(self.y_onn_test)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def plot(self, plt, model: Model, ax=None, grid_points=50, sim: bool = False):
        if ax is None:
            ax = plt.axes()
        x_min, y_min = -2.5, -2.5
        x_max, y_max = 2.5, 2.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points), np.linspace(x_min, x_max, grid_points))

        # Predict the function value for the whole grid
        inputs = []
        for x, y in zip(xx.flatten(), yy.flatten()):
            inputs.append([x, y])
        inputs = add_bias(np.asarray(inputs, dtype=np.complex64))

        Y_hat = model.predict(inputs)
        Y_hat = [yhat[0] for yhat in Y_hat]
        Z = np.array(Y_hat)
        Z = Z.reshape(xx.shape)

        # Plot the contour and training examples
        plot_handle = ax.contourf(xx, yy, Z, 50, cmap=light_rdbu, linewidths=0)
        plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], mappable=plot_handle, ax=ax)
        plot_labels(ax, dataset=self.dataset, ys=self.y_sim if sim else self.y_onn)
        plot_labels(ax, dataset=self.dataset_test, ys=self.y_sim_test if sim else self.y_onn_test)
        ax.set_ylabel(r'$x_2$', fontsize=16)
        ax.set_xlabel(r'$x_1$', fontsize=16)

    def test_accuracy(self):
        return accuracy(self.dataset_test.y, self.y_sim_test, self.y_onn_test)

    def train_accuracy(self):
        return accuracy(self.dataset.y, self.y_sim, self.y_onn)


def accuracy(actual, predicted, experimental):
    y = np.asarray([y[0] > y[1] for y in predicted], dtype=np.int32)
    y_sim = np.asarray([y[0] > y[1] for y in experimental], dtype=np.int32)
    y_act = np.asarray([y[0] > y[1] for y in actual], dtype=np.int32)
    return np.sum(np.abs(y - y_act)), np.sum(np.abs(y_sim - y_act))


def plot_labels(ax, dataset: Dataset, ys: np.ndarray):
    points_x = dataset.X.T[0, :]
    points_y = dataset.X.T[1, :]
    labels = np.array(
        [0 if yi[0] > yi[1] else 1 for yi in np.abs(ys)]).flatten()
    ax.scatter(points_x, points_y, c=labels, edgecolors='black', linewidths=0.1, s=20, cmap=dark_bwr, alpha=1)

