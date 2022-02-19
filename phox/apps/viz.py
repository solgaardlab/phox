import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

# comment out the below two lines if you have trouble getting the plots to work

import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import make_circles, make_moons, make_blobs, make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from dphox.demo import mesh
from dphox import Pattern
from scipy.special import softmax
from simphox.circuit import triangular

path_array, ps_array = mesh.demo_polys()

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

    return mpl.colors.LinearSegmentedColormap('colormap', cdict, 1024)


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


def plot_planar_boundary_from_dataset(plt, dataset, params_list, iteration, ax=None):
    if ax is None:
        ax = plt.axes()

    xx, yy, Z = get_onn_contour_data(params_list, iteration)

    # Plot the contour and training examples
    levels = np.linspace(0, 1, 50)
    plot_handle = ax.contourf(xx, yy, Z, 50, cmap=light_rdbu, linewidths=0, levels=levels)
    points_x = dataset.X.T[0, :]
    points_y = dataset.X.T[1, :]
    labels = np.array([0 if yi[0] > yi[1] else 1 for yi in np.abs(dataset.y)]).flatten()

    ax.set_ylabel(r'$x_2$', fontsize=16)
    ax.set_xlabel(r'$x_1$', fontsize=16)

    ax.scatter(points_x, points_y, c=labels, edgecolors='black', linewidths=0.1, s=10, cmap=dark_bwr)
    return plot_handle


def get_onn_contour_data(params_list, iteration, grid_points=50):
    x_min, y_min = -2.5, -2.5
    x_max, y_max = 2.5, 2.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points), np.linspace(x_min, x_max, grid_points))

    onn_layers = {'layer1': triangular(4),
                  'layer2': triangular(4),
                  'layer3': triangular(4)}
    for layer in (1, 2, 3):
        onn_layers[f'layer{layer}'].params = (
            np.mod(params_list[iteration][f'layer{layer}']['theta'], 2 * np.pi),
            np.mod(params_list[iteration][f'layer{layer}']['phi'], 2 * np.pi),
            np.mod(params_list[iteration][f'layer{layer}']['gamma'], 2 * np.pi)
        )

    # Predict the function value for the whole grid
    inputs = []
    for x, y in zip(xx.flatten(), yy.flatten()):
        inputs.append([x, y])
    inputs = add_bias(np.asarray(inputs, dtype=np.complex64))

    def predict(vin):
        out = vin
        for layer in (1, 2, 3):
            out = np.abs((onn_layers[f'layer{layer}'].matrix() @ out.T).T)
        out = softmax(np.asarray((np.sum(out[:, :2] ** 2, axis=1), np.sum(out[:, 2:] ** 2, axis=1))), axis=0)
        return out

    Y_hat = predict(inputs).T
    Y_hat = [yhat[0] for yhat in Y_hat]
    Z = np.array(Y_hat)
    Z = Z.reshape(xx.shape)

    return xx, yy, Z

def plot_amf420_powers(ax, power_data, comparison_data=None, comparison_shift=1, cmap='hot'):
    start_polys = [0, 0, 2, 4]
    normed_powers = power_data / np.sum(power_data[6])
    if comparison_data is not None:
        normed_comparison_powers = comparison_data / np.sum(comparison_data[6])

    mask = np.ones((11, 4))

    for i, poly in enumerate(start_polys):
        mask[:poly, i] = 0
        if poly > 0:
            mask[-poly:, i] = 0
        mask[poly, i] = 2
        mask[-poly - 1, i] = 3

    locs = np.array(np.where(mask == 1)).T
    left_locs = np.array(np.where(mask == 2)).T
    right_locs = np.array(np.where(mask == 3)).T

    possible_paths = path_array[::-1][:4, 4:15]

    multipolys = [possible_paths[r[1], r[0]] for r in locs]
    multipolys += [possible_paths[r[1], r[0]][2:] for r in left_locs]
    multipolys += [possible_paths[r[1], r[0]][:-1] for r in right_locs]
    waveguides = [Polygon(poly.T) for multipoly in multipolys for poly in multipoly]
    mesh = Pattern([poly for multipoly in multipolys for poly in multipoly])
    if comparison_data is not None:
        shift = (comparison_shift + 1) * mesh.size[1]
        waveguides += [Polygon(poly.T + np.hstack((np.zeros_like(poly.T[:, :1]), np.ones_like(poly.T[:, 1:]) * shift)))
                       for multipoly in multipolys for poly in multipoly]

    powers = np.hstack([[normed_powers[r[0], r[1]]] * len(mp)
                        for r, mp in zip(np.vstack((locs, left_locs, right_locs)), multipolys)])
    if comparison_data is not None:
        powers = np.hstack((powers, np.hstack([[normed_comparison_powers[r[0], r[1]]] * len(mp)
                                               for r, mp in
                                               zip(np.vstack((locs, left_locs, right_locs)), multipolys)])))
    # powers = np.sqrt(powers)

    wg_patches = PatchCollection(waveguides, cmap=cmap, lw=1, edgecolor='black')
    wg_patches.set_array(np.zeros_like(powers))
    p_patches = PatchCollection(waveguides, cmap=cmap, lw=0.5)
    p_patches.set_array(powers)
    p_patches.set_edgecolor(mpl.cm.hot(powers))
    ax.add_collection(wg_patches)
    ax.add_collection(p_patches)
    b = mesh.bounds
    ax.set_xlim(b[0] - 2, b[2] + 2)
    ax.set_ylim(b[1] - 2, b[3] * (1 + (comparison_data is not None)) + 20)
    ax.set_aspect('equal')
    p_patches.set_clim(0, 1)
    ax.text(mesh.center[0], 32, 'Measured', ha='center', va='center')
    ax.text(mesh.center[0], 64, 'Predicted', ha='center', va='center')
    return p_patches


def plot_amf420_backprop_iteration(fig, experiment: dict, iteration: int):
    subfigs = fig.subfigures(2, 1, height_ratios=[1.75, 1])
    axs = subfigs[0].subplots(4, 3)
    for layer in (1, 2, 3):
        for i in range(3):
            title = ('Forward', 'Backward', 'Sum')[i]
            power_type = ('forward', 'backward', 'sum')[i]
            ax = axs[i, layer - 1]
            data = experiment['experiment'][iteration]
            im = plot_amf420_powers(ax, data['meas'][f'{power_type}_{layer}'].squeeze()[:11, :4],
                                    np.abs(data['pred'][f'{power_type}_{layer}'].squeeze()[::2, :4]) ** 2)
            ax.set_title(f'{title}{layer}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        measured_gradients = experiment['experiment'][iteration]['meas']['gradients'][f'layer{layer}']
        predicted_gradients = experiment['experiment'][iteration]['pred']['gradients'][0][f'layer{layer}']
        ghat = np.hstack((measured_gradients['theta'][0], measured_gradients['phi'][0]))
        g = np.hstack((predicted_gradients['theta'], predicted_gradients['phi']))
        index = np.arange(len(g))
        bar_width = 0.45
        axs[-1, layer - 1].bar(index, ghat, bar_width, label=r"Measured")
        axs[-1, layer - 1].bar(index + bar_width, g, bar_width, label=r"Predicted")
        axs[-1, layer - 1].legend(bbox_to_anchor=(1, -0.01), fontsize=8)
        axs[-1, layer - 1].set_xticks([])
        axs[-1, layer - 1].set_yticks([])
        axs[-1, layer - 1].set_title(rf'Gradient{layer}: $\|g\| = {np.linalg.norm(g):3f}$')

    # cbar_ax = subfigs[0].add_axes([1, 0.25, 0.02, 0.7])
    # fig.colorbar(im, cax=cbar_ax, label='Waveguide power')
    dataset = Dataset(experiment['X_train'], experiment['y_train'])
    axs = subfigs[1].subplots(1, 2)
    axs[0].plot([experiment['experiment'][i]['train_loss'] for i in range(1000)], label='train', color='black',
                linestyle='dotted')
    axs[0].plot([experiment['experiment'][i]['test_loss'] for i in range(1000)], label='test', color='black')
    axs[0].scatter(iteration, experiment['experiment'][iteration]['train_loss'], color='black')
    axs[0].scatter(iteration, experiment['experiment'][iteration]['test_loss'], color='black')
    axs[0].legend()
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Cost function $\mathcal{L}$')
    plot_handle = plot_planar_boundary_from_dataset(plt, dataset, experiment['params_list'], iteration, ax=axs[1])
    plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], mappable=plot_handle, ax=axs[1])
