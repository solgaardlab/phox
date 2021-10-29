from ..typing import Callable
from .viz import Dataset, add_bias, light_rdbu, dark_bwr
from ..experiment import Sputnik

import numpy as np
from tensorflow.keras import Model
from .viz import plot_planar_boundary
from scipy.special import softmax

import time
import pickle


#TODO(sunil): fix a lot of boilerplate code.
class ONN2D:
    def __init__(self, dataset: Dataset, dataset_test: Dataset,
                 n_layers: int, thetas: list = None, phis: list = None,
                 y_sim: np.ndarray = None, y_sim_test: np.ndarray = None,
                 y_onn: np.ndarray = None, y_onn_test: np.ndarray = None):
        self.n_layers = n_layers
        self.thetas = thetas
        self.phis = phis
        self.dataset = dataset
        self.dataset_test = dataset_test
        self.y_sim = [] if y_sim is None else np.asarray(y_sim)
        self.y_sim_test = [] if y_sim_test is None else np.asarray(y_sim_test)
        self.y_onn = [] if y_onn is None else y_onn
        self.y_onn_test = [] if y_onn_test is None else y_onn_test

    def self_configure(self, mesh: Sputnik, model, pbar: Callable):
        thetas = []
        phis = []

        for i in pbar(range(self.n_layers)):
            mesh.set_unitary_sc(model.layers[i].matrix.conj())
            ts, ps = mesh.get_unitary_phases()
            thetas.append(ts)
            phis.append(ps)
        mesh.to_layer(16)  # the layer where outputs are measured
        self.thetas = thetas
        self.phis = phis

    def onn(self, input_vector: np.ndarray, mesh: Sputnik,
            meas_delay: float = 1, factor: float = 3):
        outputs = input_vector
        for ts, ps in zip(self.thetas, self.phis):
            mesh.set_input(np.hstack((outputs, 0)))
            mesh.set_unitary_phases(ts, ps)
            time.sleep(meas_delay)
            spots = np.abs(mesh.camera.spot_powers[::3])
            outputs = (np.sqrt(spots / np.sum(np.abs(spots))))[:4]
        return softmax(factor ** 2 * np.asarray((np.sum(outputs[:2] ** 2), np.sum(outputs[2:] ** 2))))

    def classify_train(self, mesh, model, pbar=None, meas_time: float = 1):
        self.y_sim = []
        self.y_onn = []
        iterator = pbar(range(len(self.dataset.X))) if pbar else range(len(self.dataset.X))
        for i in iterator:
            self.y_sim.append(model(self.dataset.X[i]))
            self.y_onn.append(self.onn(self.dataset.X[i], mesh, meas_time))
        self.y_sim = np.asarray(self.y_sim)[:, 0, :]
        self.y_onn = np.asarray(self.y_onn)

    def classify_test(self, mesh, model, pbar=None, meas_time: float = 1):
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

    def plot(self, plt, model, ax=None, grid_points=50, sim: bool = False):
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


def accuracy(actual, predicted, simulated):
    y = np.asarray([y[0] > y[1] for y in predicted], dtype=np.int)
    y_sim = np.asarray([y[0] > y[1] for y in simulated], dtype=np.int)
    y_act = np.asarray([y[0] > y[1] for y in actual], dtype=np.int)
    return np.sum(np.abs(y - y_act)), np.sum(np.abs(y_sim - y_act))


def plot_labels(ax, dataset: Dataset, ys: np.ndarray):
    points_x = dataset.X.T[0, :]
    points_y = dataset.X.T[1, :]
    labels = np.array(
        [0 if yi[0] > yi[1] else 1 for yi in np.abs(ys)]).flatten()
    ax.scatter(points_x, points_y, c=labels, edgecolors='black', linewidths=0.1, s=20, cmap=dark_bwr, alpha=1)
