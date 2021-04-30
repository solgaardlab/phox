import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

# comment out the below two lines if you have trouble getting the plots to work

import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import make_circles, make_moons, make_blobs, make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

Dataset = namedtuple('Dataset', ['X', 'y'])


def add_bias(x, p=9, n=4):
    abs_sq_term = np.sum(x ** 2, axis=1)[:, np.newaxis]
    abs_sq_term[abs_sq_term > p] = p  # avoid nan in case it exists
    normalized_bias_term = np.sqrt(p - abs_sq_term)
    return np.hstack([x, normalized_bias_term / np.sqrt(n - 2) * np.ones((x.shape[0], n - 2), dtype=np.complex64)])


def get_planar_dataset_with_circular_bias(dataset_name, test_size=.2):
    if dataset_name == 'moons':
        X, y = make_moons(noise=0.2, random_state=0, n_samples=250)
    elif dataset_name == 'circle':
        X, y = make_circles(noise=0.2, factor=0.5, random_state=1, n_samples=250)
    elif dataset_name == 'blobs':
        X, y = make_blobs(random_state=5, n_features=2,
                          centers=[(-0.4, -0.4), (0.25, 0.3)],
                          cluster_std=0.5, n_samples=250)
    elif dataset_name == 'ring':
        X, y = make_gaussian_quantiles(n_features=2, n_classes=3, n_samples=500, cov=0.4)
        y[y == 2] = 0
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train_f = add_bias(X_train).astype(np.complex64)
    X_test_f = add_bias(X_test).astype(np.complex64)
    y_train_f = y_train.astype(np.complex64)
    y_test_f = y_test.astype(np.complex64)

    return Dataset(X, y), Dataset(X_train_f, y_train_f), Dataset(X_test_f, y_test_f)


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red', 'green', 'blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1],), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


light_rdbu = cmap_map(lambda x: 0.5 * x + 0.5, plt.cm.RdBu)
dark_bwr = cmap_map(lambda x: 0.75 * x, plt.cm.bwr)


def plot_planar_boundary(plt, dataset, model, ax=None, grid_points=50):
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
    points_x = dataset.X.T[0, :]
    points_y = dataset.X.T[1, :]
    labels = np.array([0 if yi[0] > yi[1] else 1 for yi in np.abs(dataset.y)]).flatten()

    ax.set_ylabel(r'$x_2$', fontsize=16)
    ax.set_xlabel(r'$x_1$', fontsize=16)

    ax.scatter(points_x, points_y, c=labels, edgecolors='black', linewidths=0.1, s=10, cmap=dark_bwr)
