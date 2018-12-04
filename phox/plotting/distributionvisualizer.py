from typing import List, Callable

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .helpers import get_theta_distribution, get_xi_distribution,\
    get_simulated_theta_histogram_from_xi, get_simulated_uniform_phase_histogram,\
    get_simulated_xi_histogram_from_theta, transmittivity_haar_phase


class DistributionVisualizer:
    def __init__(self, alphas: List[int]):
        self.alphas = alphas

    def plot_theta_xi(self, ax, num_thetas: int=10000):
        theta = np.linspace(0, np.pi / 2, num_thetas)
        for alpha in self.alphas:
            ax.plot(theta, np.power(np.sin(theta), 2 * alpha))
        ax.set_xlabel(r'Phase ($\theta / 2$)')
        ax.set_ylabel(r'Haar Phase ($\xi_\alpha$)')
        ax.set_xticks([0, np.pi / 4, np.pi / 2])
        ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'])
        ax.legend([rf'$\alpha = {alpha}$' for alpha in self.alphas], fontsize=9)

    def plot_xi_theta(self, ax, num_thetas: int=10000):
        theta = np.linspace(0, np.pi / 2, num_thetas)
        for alpha in self.alphas:
            ax.plot(np.power(np.sin(theta), 2 * alpha), theta)
        ax.set_ylabel(r'Phase ($\theta / 2$)')
        ax.set_xlabel(r'Haar Phase ($\xi_\alpha$)')
        ax.set_yticks([0, np.pi / 4, np.pi / 2])
        ax.set_yticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'])
        ax.legend([rf'$\alpha = {alpha}$' for alpha in self.alphas], fontsize=9)

    def plot_simulated_theta_distributions(self, ax, simulate: bool=False, num_samples: int=1e6, num_bins: int=1000):
        uniform_xi = np.random.rand(int(num_samples))
        inset_ax = inset_axes(ax,
                              height="30%",  # set height
                              width="30%",  # and width
                              loc=10)

        xis, xi_freqs = get_simulated_uniform_phase_histogram(num_samples, num_bins, haar_phase=True) if simulate\
            else get_xi_distribution(num_xis=num_samples, override_uniform=True)
        inset_ax.plot(xis, xi_freqs, color='black')
        inset_ax.set_ylim(0, 2)
        inset_ax.set_xlabel(r'Haar Phase ($\xi_\alpha$)')
        inset_ax.get_yaxis().set_ticks([])

        for alpha in self.alphas:
            thetas, theta_freqs = get_simulated_theta_histogram_from_xi(uniform_xi, alpha, num_bins) if simulate else \
                get_theta_distribution(alpha=alpha, num_thetas=num_samples)
            ax.plot(thetas, theta_freqs)
        ax.set_xticks([0, np.pi / 2, np.pi])
        ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'])
        ax.set_xlabel(r'Phase ($\theta / 2$)')
        ax.legend([rf'$\alpha = {alpha}$' for alpha in self.alphas], fontsize=9)
        ax.get_yaxis().set_ticks([])

    def plot_simulated_xi_distributions(self, ax, simulate: bool=False, num_samples: int=1e6, num_bins: int=1000):
        # still needs more work (to get log xi predicted, currently it is simulated)
        uniform_theta = np.pi / 2 * np.random.rand(int(num_samples))

        inset_ax = inset_axes(ax,
                              height="30%",  # set height
                              width="30%",  # and width
                              loc=9
                              )

        thetas, theta_freqs = get_simulated_uniform_phase_histogram(num_samples, num_bins) if simulate\
            else get_theta_distribution(num_thetas=num_samples, override_uniform=True)
        inset_ax.plot(thetas, theta_freqs, color='black')
        inset_ax.set_xticks([0, np.pi / 2, np.pi])
        inset_ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'])
        inset_ax.set_xlabel(r'Phase ($\theta / 2$)')

        for alpha in self.alphas:
            xis, xi_freqs = get_simulated_xi_histogram_from_theta(uniform_theta, alpha, num_bins) if simulate else \
                get_xi_distribution(alpha=alpha, num_xis=num_samples)
            ax.plot(xis, xi_freqs)
            ax.set_xlabel(r'Haar Phase ($\xi_\alpha$)')
        ax.get_yaxis().set_ticks([])
        inset_ax.get_yaxis().set_ticks([])
        inset_ax.set_ylim(0, 4 / np.pi)
        ax.legend([rf'$\alpha = {alpha}$' for alpha in self.alphas], fontsize=9)

    def plot_haar_transmittivities(self, ax, num_samples: int=10000):
        xi = np.linspace(-2, 2, num_samples)
        for alpha in self.alphas:
            ax.plot(xi, transmittivity_haar_phase(xi, alpha))
        ax.set_xlabel(r'Haar Phase ($\xi_\alpha$)')
        ax.set_ylabel(r'Transmission Coefficient ($t = \sin(\theta(\xi_\alpha))$)')
        ax.legend([rf'$\alpha = {alpha}$' for alpha in self.alphas], fontsize=9)

    @staticmethod
    def plot_theta_variances(ax, alphas: List[int], pbar_handle: Callable=None):
        uniform_xi = np.random.rand(1000000)
        iterator = pbar_handle(alphas) if pbar_handle else alphas
        stds = [np.std(np.arcsin(np.power(uniform_xi, 1 / (2 * alpha)))) for alpha in iterator]
        ax.plot(alphas, np.log(stds))
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'Log Standard Deviation ($\log_{10} \sigma_{\theta; \alpha}$)')
