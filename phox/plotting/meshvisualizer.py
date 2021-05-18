from typing import List, Optional

from functools import partial

from .helpers import *
from neurophox.helpers import get_alpha_checkerboard
from matplotlib.patches import Polygon

LABELLER_TO_LABEL_FUNC = {
    'downright_diagonal_layer': make_downright_diagonal_layer_labels,
    'upright_diagonal_layer': make_upright_diagonal_layer_labels,
    'tmn': make_givens_rotation_tmn_labels,
    'xy': make_xy_diagonal_labels,
    'io': make_input_output_labels,
    'vertical_layer': make_vertical_layer_labels,
    'transmission_mzi': partial(make_theta_mzi_labels, mean=False),
    'theta_mzi': partial(make_theta_mzi_labels, mean=False),
    'exp_transmission_mzi': partial(make_transmission_mzi_labels, mean=True),
    'exp_theta_mzi': partial(make_theta_mzi_labels, mean=True),
    'tri_diagonal_layer': make_tri_diagonal_layer_labels,
    'tri_vertical_layer': make_tri_vertical_layer_labels,
    'tri_tmn': make_tri_givens_rotation_tmn_labels,
    'tri_transmission_mzi': partial(make_tri_transmission_mzi_labels, mean=False),
    'tri_theta_mzi': partial(make_tri_theta_mzi_labels, mean=False),
    'exp_tri_transmission_mzi': partial(make_tri_transmission_mzi_labels, mean=True),
    'exp_tri_theta_mzi': partial(make_tri_theta_mzi_labels, mean=True)
}


# TODO (sunil): this whole class/file will need a rewrite eventually.
class MeshVisualizer:
    def __init__(self, dim: int=8, label_fontsize=10,
                 transformer_name: str='rd', use_double_size: bool=False, plt=None):
        self.dim = dim
        self.label_fontsize = label_fontsize
        self.transformer_name = transformer_name
        self.use_double_size = use_double_size
        self.marker_size = int(2.5 * self.label_fontsize)
        self.plt = plt

    def plot_model_figure(self, ax, labellers: Optional[List]=None, include_alpha=False, include_smn=False):
        if self.transformer_name == 'cgrd':
            self._plot_cg_model(ax)
        elif self.transformer_name == 'rrd':
            self._plot_rrd_model(ax)
        elif self.transformer_name == 'tri':
            self._plot_triangular_model(ax, labellers, include_alpha)
        else:
            self._plot_local_model(ax, labellers, include_alpha, include_smn)

    def _plot_rectangular_mzi_lines(self, ax, points):
        input_locations = np.arange(self.dim) + 0.5
        rank = max(points[0])

        lines_to_plot = []
        for point in np.asarray(points).T:
            if point[0] == rank:
                lines_to_plot.append(output_piece(point, upper=False, rank=rank))
                lines_to_plot.append(output_piece(point, upper=True, rank=rank))
            else:
                if point[0] == 1:
                    lines_to_plot.append(input_piece(point, upper=False))
                    lines_to_plot.append(input_piece(point, upper=True))
                if point[0] == 2 and point[1] == self.dim - 1 and self.dim % 2:
                    lines_to_plot.append(input_piece(point, upper=True))
                if point[1] == 1:
                    if point[0] + 2 > rank:
                        lines_to_plot.append(output_piece(point, upper=False, rank=rank))
                    else:
                        lines_to_plot.append(lower_bounce(point))
                else:
                    lines_to_plot.append(forward_connect(point, upper=False))
                if point[1] == self.dim - 1:
                    if point[0] + 2 > rank:
                        lines_to_plot.append(output_piece(point, upper=True, rank=rank))
                    else:
                        lines_to_plot.append(upper_bounce(point, dim=self.dim))
                else:
                    lines_to_plot.append(forward_connect(point, upper=True))

        for line in lines_to_plot:
            ax.plot(line[0], line[1], color='black', zorder=1, linewidth=1)

        ax.scatter(-np.ones(self.dim), input_locations,
                   color=[0, 0.2, 0.6], marker='>', zorder=2, s=self.marker_size)
        ax.scatter((rank + 2) * np.ones(self.dim), input_locations,
                   color=DARK_PURPLE, marker='s', zorder=2, s=self.marker_size)

    def _plot_triangular_mzi_lines(self, ax, points, add_light_prop=False):
        input_locations = np.arange(self.dim) + 0.5
        rank = self.dim * 2 - 3

        lines_to_plot = []
        for point in np.asarray(points).T:
            if point[0] == self.dim - point[1]:
                lines_to_plot.append(input_piece(point, upper=False))
                if point[1] == self.dim - 1:
                    lines_to_plot.append(input_piece(point, upper=True))
            if point[0] == self.dim - 2 + point[1]:
                lines_to_plot.append(output_piece(point, upper=False, rank=rank))
                if point[1] == self.dim - 1:
                    lines_to_plot.append(output_piece(point, upper=True, rank=rank))
                else:
                    lines_to_plot.append(forward_connect(point, upper=True))
            else:
                lines_to_plot.append(forward_connect(point, upper=False))
                if point[1] == self.dim - 1:
                    lines_to_plot.append(upper_bounce(point, dim=self.dim))
                else:
                    lines_to_plot.append(forward_connect(point, upper=True))

        for line in lines_to_plot:
            ax.plot(line[0], line[1], color='black', zorder=1, linewidth=1)

        if add_light_prop:
            blue_lines_to_plot = []
            purple_lines_to_plot = []
            red_lines_to_plot = []
            for point in np.asarray(points).T:
                if point[0] - point[1] == self.dim - 2:
                    if point[0] == self.dim - 1:
                        blue_lines_to_plot.append(input_piece(point, upper=False))
                    purple_lines_to_plot.append(output_piece(point, upper=False, rank=rank))
                    if point[1] == self.dim - 1:
                        purple_lines_to_plot.append(output_piece(point, upper=True, rank=rank))
                    else:
                        red_lines_to_plot.append(forward_connect(point, upper=True))
                    ax.annotate(r'$\boldsymbol{T}_{'+str(point[1])+'}$', (point[0] + 0.25, point[1] + 0.75),
                                color=DARK_RED, fontsize=int(self.label_fontsize / 2),
                                horizontalalignment='center', verticalalignment='center')
                    ax.annotate(r'$x_{' + str(point[1]) + '}$', (rank + 2.5, point[1] - 0.5),
                                color=DARK_PURPLE, fontsize=int(self.label_fontsize),
                                horizontalalignment='left', verticalalignment='center')
                    ax.annotate(r'$1 - T_{'+str(point[1])+'}$', (point[0] + 0.5, point[1] - 1),
                                color=DARK_RED, fontsize=int(self.label_fontsize / 2),
                                horizontalalignment='left', verticalalignment='top')

            for line in blue_lines_to_plot:
                ax.plot(line[0], line[1], color=DARK_BLUE, zorder=1, linewidth=2)
            for line in red_lines_to_plot:
                ax.plot(line[0], line[1], color='black', zorder=1, linewidth=2)
            for line in purple_lines_to_plot:
                ax.plot(line[0], line[1], color=DARK_PURPLE, zorder=1, linewidth=2)

        ax.scatter(-np.ones(self.dim), input_locations,
                   color=[0, 0.2, 0.6], marker='>', zorder=2, s=self.marker_size)
        ax.scatter((rank + 2) * np.ones(self.dim), input_locations,
                   color=DARK_PURPLE, marker='s', zorder=2, s=self.marker_size)

    def _plot_local_model(self, ax, labellers: Optional[List], include_alpha: bool, include_smn: bool, rank: int=None):
        tunable_dim = int(self.dim / 2) if self.use_double_size else self.dim
        rank = rank if rank is not None else tunable_dim
        checkerboard_points = get_checkerboard_points(tunable_dim, rank)
        points = checkerboard_points[1] + 1, checkerboard_points[0] + 1
        self._plot_rectangular_mzi_lines(ax, points)
        ax.scatter(*points, color=DARK_RED, zorder=2, s=self.marker_size)
        if include_alpha:
            alpha_checkerboard = get_alpha_checkerboard(tunable_dim, rank)
            alphas = alpha_checkerboard[checkerboard_points]
            for i in range(len(checkerboard_points[0])):
                point = (checkerboard_points[1][i] + 1, checkerboard_points[0][i] + 0.5)
                ax.annotate(rf'${int(alphas[i])}$', point, color=DARK_RED, fontsize=int(3 * self.label_fontsize / 4),
                            horizontalalignment='center', verticalalignment='center')
        if include_smn:
            smn_checkerboard = get_smn_checkerboard(tunable_dim, tunable_dim)
            smns = smn_checkerboard[checkerboard_points]
            for i in range(len(checkerboard_points[0])):
                point = (checkerboard_points[1][i] + 1, checkerboard_points[0][i] + 1.5)
                ax.annotate(rf'${int(smns[i])}$', point, color=DARK_BLUE, fontsize=int(self.label_fontsize / 2),
                            horizontalalignment='center', verticalalignment='center')
        if labellers is None:
            labellers = ['io', 'tmn', 'vertical_layer']
        for labeller in labellers:
            LABELLER_TO_LABEL_FUNC[labeller](ax, tunable_dim, self.label_fontsize)
        if include_alpha or include_smn:
            ax.scatter([rank + 2 + tunable_dim], [tunable_dim], color=DARK_RED, zorder=2, s=self.marker_size)
        if include_alpha:
            ax.annotate(r'$\alpha_{n\ell}$', (rank + 2 + tunable_dim, tunable_dim - 0.5),
                        color=DARK_RED, fontsize=int(3 * self.label_fontsize / 4),
                        horizontalalignment='center', verticalalignment='center')
        if include_smn:
            ax.annotate(r'$s_{x}[y]$', (rank + 2 + tunable_dim, tunable_dim + 0.5),
                        color=DARK_BLUE, fontsize=int(self.label_fontsize / 2),
                        horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_ylim(tunable_dim + 2, -3)

    def _plot_triangular_model(self, ax, labellers: Optional[List], include_alpha: bool):
        orig_points = get_triangular_mesh_points(self.dim)
        points = orig_points[1] - 1 + self.dim % 2, orig_points[0] + 1
        self._plot_triangular_mzi_lines(ax, points, add_light_prop=labellers and 'light_prop' in labellers)
        if include_alpha:
            for point in np.asarray(points).T:
                label_point = (point[0], point[1] - 0.5)
                ax.annotate(rf'${self.dim - point[1]}$', label_point, color=DARK_RED,
                            fontsize=int(3 * self.label_fontsize / 4),
                            horizontalalignment='center', verticalalignment='center')
            ax.scatter([self.dim * 2 + 3], [self.dim / 2], color=DARK_RED, zorder=2, s=self.marker_size)
            ax.annotate(r'$\alpha_{n\ell}$', (self.dim * 2 + 3, self.dim / 2 - 0.5),
                        color=DARK_RED, fontsize=int(3 * self.label_fontsize / 4),
                        horizontalalignment='center', verticalalignment='center')
        if labellers is None:
            labellers = ['io', 'tri_tmn', 'tri_vertical_layer']
        for labeller in labellers:
            if labeller == 'light_prop':
                continue
            LABELLER_TO_LABEL_FUNC[labeller](ax, self.dim, self.label_fontsize)
        ax.scatter(points[0], points[1], color=DARK_RED, zorder=2, s=self.marker_size)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_ylim(self.dim + 1, -1)

    def _plot_rrd_model(self, ax, embed=False):
        half_checkerboard_points = get_checkerboard_points(self.dim, self.dim)
        checkerboard_points = get_checkerboard_points(self.dim, self.dim * 2)
        if embed:
            canvas_checkerboard_points = get_checkerboard_points(self.dim * 2, self.dim * 2)
            points = canvas_checkerboard_points[1] + 1, canvas_checkerboard_points[0] + 1
            ax.scatter(canvas_checkerboard_points[1] + 1,
                       canvas_checkerboard_points[0] + 1, color='black', zorder=2, s=self.marker_size)
        else:
            points = checkerboard_points[1] + 1, checkerboard_points[0] + 1
        self._plot_rectangular_mzi_lines(ax, points)
        ax.scatter(half_checkerboard_points[1] + 1,
                   half_checkerboard_points[0] + 1, color=DARK_RED, zorder=2, s=self.marker_size)
        ax.scatter(half_checkerboard_points[1] + 1 + self.dim,
                   half_checkerboard_points[0] + 1, color=DARK_GREEN, zorder=2, s=self.marker_size)
        for i in range(self.dim + 1):
            ax.plot([i + 0.5, i + 0.5], [0, self.dim], color=DARK_RED, alpha=0.5)
            if i < self.dim:
                ax.annotate(
                    f'${i + 1}$', xy=(i + 1, -0.5), fontsize=self.label_fontsize,
                    horizontalalignment='center', verticalalignment='center', color=DARK_RED
                )
        for i in range(self.dim, self.dim * 2 + 1):
            ax.plot([i + 0.5, i + 0.5], [0, self.dim], color=DARK_GREEN, alpha=0.5)
            if i < self.dim * 2:
                ax.annotate(
                    f'${i + 1}$', xy=(i + 1, -0.5), fontsize=self.label_fontsize,
                    horizontalalignment='center', verticalalignment='center', color=DARK_GREEN
                )
        for i in range(self.dim - 1):
            ax.plot([1 + i % 2, 2 * self.dim + 3], [i + 1, i + 1], color=DARK_ORANGE, alpha=0.5, zorder=1, linestyle='--')
            idx_str = f'{i + 1}'
            ax.annotate(
                '$U_{'+idx_str+'}$', xy=(2 * self.dim + 3.25, i + 1), fontsize=self.label_fontsize,
                horizontalalignment='left', verticalalignment='center', color=DARK_ORANGE
            )
        for i in range(self.dim):
            ax.annotate(
                f'${i + 1}$', xy=(-1.5, i + 0.5), fontsize=self.label_fontsize,
                horizontalalignment='right', verticalalignment='center', color=[0, 0.2, 0.6]
            )
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_ylim(self.dim + 1, -1)
        plt.tight_layout()

    def _plot_cg_model(self, ax, embed=False):
        self.dim = self.dim
        tunable_ranks, sampling_frequencies = get_default_coarse_grain_ranks(self.dim, use_cg_sequence=False)
        current_start_index = 0
        start_indices = []
        end_indices = []
        sampling_checkerboard_points_list = []
        for i in range(int(np.floor(np.log2(self.dim))) - 1):
            current_start_index += np.floor(self.dim / np.log2(self.dim))
            start_indices.append(current_start_index)
            sampling_checkerboard_points_list.append(get_checkerboard_points(self.dim, sampling_frequencies[i]))
            current_start_index += sampling_frequencies[i]
            end_indices.append(current_start_index)

        checkerboard_points = get_checkerboard_points(self.dim, sum(tunable_ranks) + sum(sampling_frequencies))
        if embed:
            canvas_checkerboard_points = get_checkerboard_points(2 * self.dim, 2 * self.dim)
            points = canvas_checkerboard_points[1] + 1, canvas_checkerboard_points[0] + 1
            ax.scatter(canvas_checkerboard_points[1] + 1,
                       canvas_checkerboard_points[0] + 1, color='black', zorder=2, s=self.marker_size)
        else:
            points = checkerboard_points[1] + 1, checkerboard_points[0] + 1

        self._plot_rectangular_mzi_lines(ax, points)
        ax.scatter(checkerboard_points[1] + 1, checkerboard_points[0] + 1, color=DARK_RED, zorder=2, s=self.marker_size)

        idx = 0
        layer_idx = 0

        for start_idx, end_idx, checkerboard_points in zip(start_indices, end_indices, sampling_checkerboard_points_list):
            prev_index = 0 if idx == 0 else end_indices[idx - 1]
            for i in range(int(prev_index), int(start_idx) + 1):
                ax.plot([i + 0.5, i + 0.5], [0, self.dim - 0.5], color=DARK_RED, alpha=0.5)
                if i > prev_index:
                    layer_idx += 1
                    ax.annotate(
                        f'${layer_idx}$', xy=[i, -0.5], fontsize=self.label_fontsize,
                        horizontalalignment='center', verticalalignment='center', color=DARK_RED
                    )
            ax.annotate(
                f'$M_{idx + 1}$', xy=((start_idx + prev_index) / 2 + 0.5, -1.5), fontsize=self.label_fontsize * 1.2,
                horizontalalignment='center', verticalalignment='center', color=DARK_RED
            )
            ax.annotate(
                f'$P_{idx + 1}$', xy=((start_idx + end_idx) / 2 + 0.5, -1), fontsize=self.label_fontsize * 1.2,
                horizontalalignment='center', verticalalignment='center', color=GRAY
            )
            ax.scatter(checkerboard_points[1] + 1 + start_idx,
                       checkerboard_points[0] + 1, color=GRAY, zorder=2, s=self.marker_size)
            idx += 1
        for i in range(int(current_start_index), int(current_start_index) + 5):
            ax.plot([i + 0.5, i + 0.5], [0, self.dim - 0.5], color=DARK_RED, alpha=0.5)
            if i > current_start_index:
                layer_idx += 1
                ax.annotate(
                    f'${layer_idx}$', xy=[i, -0.5], fontsize=self.label_fontsize,
                    horizontalalignment='center', verticalalignment='center', color=DARK_RED
                )
        ax.annotate(
            f'$M_{idx + 1}$', xy=(current_start_index + 2.5, -1.5), fontsize=self.label_fontsize * 1.2,
            horizontalalignment='center', verticalalignment='center', color=DARK_RED
        )
        for i in range(self.dim - 1):
            ax.plot([1 + i % 2, 2 * self.dim+1], [i + 1, i + 1], color=DARK_ORANGE, alpha=0.5, zorder=1, linestyle='--')
            idx_str = f'{i + 1}'
            ax.annotate(
                '$U_{'+idx_str+'}$', xy=(2 * self.dim + 1.25, i + 1), fontsize=self.label_fontsize,
                horizontalalignment='left', verticalalignment='center', color=DARK_ORANGE
            )
        for i in range(self.dim):
            ax.annotate(
                f'${i + 1}$', xy=(-1.5, i + 0.5), fontsize=self.label_fontsize,
                horizontalalignment='right', verticalalignment='center', color=DARK_BLUE
            )
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_ylim(self.dim + 1, -1)
        plt.tight_layout()

    def _plot_binary_tree(self, ax):

        def add_appendage(start, input: bool, upper: bool, opaque=False):
            point_0 = (start[0] + (2 * input - 1), start[1] + (1 - 2 * upper))
            point_1 = (point_0[0] + (2 * input - 1) * 1.5, point_0[1])
            alpha = 1 - (opaque + 1) / 2
            ax.plot((start[0], point_0[0]), (start[1], point_0[1]),
                    color=(alpha, alpha, alpha), zorder=1)
            ax.plot((point_0[0], point_1[0]), (point_0[1], point_1[1]),
                    color=(alpha, alpha, alpha), zorder=1)
            ax.scatter(point_1[0], point_1[1],
                       color=DARK_PURPLE if opaque else LIGHT_PURPLE, marker='s' if opaque else 'x', zorder=2,
                       s=2 * self.marker_size)

        if self.dim == 4:
            points_list = [(0, 0), (2, 2), (2, -2)]
            input_locations = [(4.5, 3), (4.5, -3), (4.5, 1), (4.5, -1)]
            ax.plot((0, 3), (0, 3), color='black', zorder=1)
            ax.plot((0, 3), (0, -3), color='black', zorder=1)
            ax.plot((2, 3), (2, 1), color='black', zorder=1)
            ax.plot((2, 3), (-2, -1), color='black', zorder=1)
            for point in input_locations:
                ax.plot((point[0] - 1.5, point[0]), (point[1], point[1]), color='black', zorder=1)
                ax.scatter(*point, color=[0, 0.2, 0.6], marker='>', zorder=2, s=2 * self.marker_size)
            for point in points_list:
                ax.scatter(*point, color=DARK_RED, zorder=2, s=2 * self.marker_size)
        elif self.dim == 8:
            points_list = [(0, 0), (4, -4), (4, 4), (6, 2), (6, 6), (6, -6), (6, -2)]
            alpha_beta_list = [(4, 4), (2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1)]
            labels = [r"$\boldsymbol{T}_7$", r"$\boldsymbol{T}_5$", r"$\boldsymbol{T}_6$",
                      r"$\boldsymbol{T}_2$", r"$\boldsymbol{T}_1$", r"$\boldsymbol{T}_4$", r"$\boldsymbol{T}_3$"]
            input_locations = [(8.5, 7), (8.5, -7), (8.5, 5), (8.5, -5), (8.5, 3), (8.5, -3), (8.5, 1), (8.5, -1)]
            ax.plot((0, 7), (0, 7), color='black', zorder=1)
            ax.plot((0, 7), (0, -7), color='black', zorder=1)
            ax.plot((6, 7), (6, 5), color='black', zorder=1)
            ax.plot((6, 7), (-6, -5), color='black', zorder=1)
            ax.plot((6, 7), (2, 3), color='black', zorder=1)
            ax.plot((6, 7), (-2, -3), color='black', zorder=1)
            ax.plot((4, 7), (-4, -1), color='black', zorder=1)
            ax.plot((4, 7), (4, 1), color='black', zorder=1)
            for point in input_locations:
                ax.plot((point[0] - 1.5, point[0]), (point[1], point[1]), color='black', zorder=1)
                ax.scatter(*point, color=DARK_GREEN, marker='>', zorder=2, s=2 * self.marker_size)
            flip_appendage = False
            for label, alpha_beta, point in zip(labels, alpha_beta_list, points_list):
                alpha, beta = alpha_beta
                ax.annotate(
                    r"$"+str(beta)+"$", xy=(point[0], point[1] + 0.4), fontsize=self.label_fontsize,
                    horizontalalignment='center', verticalalignment='bottom', color=DARK_BLUE
                )
                ax.annotate(
                    r"$"+str(alpha)+"$", xy=(point[0], point[1] - 0.4), fontsize=self.label_fontsize,
                    horizontalalignment='center', verticalalignment='top', color=DARK_RED
                )
                ax.annotate(
                    label, xy=(point[0] + 0.5, point[1]), fontsize=self.label_fontsize * 1.2,
                    horizontalalignment='right', verticalalignment='center', color=DARK_RED
                )
                ax.scatter(*point, color=DARK_RED, zorder=2, s=2 * self.marker_size)
                add_appendage(point, input=False, upper=flip_appendage)
                flip_appendage = not flip_appendage
            ax.plot((0, -6), (0, -6), color='black', zorder=1)
            add_appendage((-6, -6), input=False, upper=True, opaque=True)

            lpoint = (-8, 0)

            ax.annotate(
                r"$\boldsymbol{T}_n$", xy=(lpoint[0] + 0.5, lpoint[1]), fontsize=self.label_fontsize * 1.2,
                horizontalalignment='right', verticalalignment='center', color=DARK_RED
            )
            ax.annotate(
                r"$\beta_n$", xy=(lpoint[0], lpoint[1] + 0.4), fontsize=self.label_fontsize,
                horizontalalignment='center', verticalalignment='bottom', color=DARK_BLUE
            )
            ax.annotate(
                r"$\alpha_n$", xy=(lpoint[0], lpoint[1] - 0.4), fontsize=self.label_fontsize,
                horizontalalignment='center', verticalalignment='top', color=DARK_RED
            )
            ax.scatter(*lpoint, color=DARK_RED, zorder=2, s=2 * self.marker_size)
        else:
            raise NotImplementedError("Sad")

    def _plot_linear_chain(self, ax):

        def add_appendage(start, input: bool, upper: bool, opaque=False):
            point_0 = (start[0] + (2 * input - 1), start[1] + (1 - 2 * upper))
            point_1 = (point_0[0] + (2 * input - 1) * 1.5, point_0[1])
            alpha = 1 - (opaque + 1) / 2
            ax.plot((start[0], point_0[0]), (start[1], point_0[1]),
                    color=(alpha, alpha, alpha), zorder=1)
            ax.plot((point_0[0], point_1[0]), (point_0[1], point_1[1]),
                    color=(alpha, alpha, alpha), zorder=1)
            ax.scatter(point_1[0], point_1[1],
                       color=DARK_PURPLE if opaque else LIGHT_PURPLE, marker='s' if opaque else 'x', zorder=2,
                       s=2 * self.marker_size)

        if self.dim == 4:
            points_list = [(0, 0), (2, 2), (-2, -2)]
            input_locations = [(4.5, 3), (4.5, -3), (4.5, 1), (4.5, -1)]
            ax.plot((0, 3), (0, 3), color='black', zorder=1)
            ax.plot((0, 3), (0, -3), color='black', zorder=1)
            ax.plot((2, 3), (2, 1), color='black', zorder=1)
            ax.plot((2, 3), (-2, -1), color='black', zorder=1)
            for point in input_locations:
                ax.plot((point[0] - 1.5, point[0]), (point[1], point[1]), color='black', zorder=1)
                ax.scatter(*point, color=DARK_GREEN, marker='>', zorder=2, s=2 * self.marker_size)
            for point in points_list:
                ax.scatter(*point, color=DARK_RED, zorder=2, s=2 * self.marker_size)
        elif self.dim == 8:
            points_list = [(0, 0), (-4, -4), (4, 4), (2, 2), (-2, -2), (6, 6), (-6, -6)]
            labels = [r"$\boldsymbol{T}_4$", r"$\boldsymbol{T}_6$", r"$\boldsymbol{T}_2$", r"$\boldsymbol{T}_3$",
                      r"$\boldsymbol{T}_5$", r"$\boldsymbol{T}_1$", r"$\boldsymbol{T}_7$"]
            alpha_beta_list = [(4, 1), (6, 1), (2, 1), (3, 1), (5, 1), (1, 1), (7, 1)]
            input_locations = [(8.5, 7), (8.5, 5), (8.5, 3), (8.5, 1),
                               (8.5, -1), (8.5, -3), (8.5, -5), (8.5, -7)]
            ax.plot((0, 1), (0, -1), color='black', zorder=1)
            ax.plot((2, 3), (2, 1), color='black', zorder=1)
            ax.plot((-2, -1), (-2, -3), color='black', zorder=1)
            ax.plot((4, 5), (4, 3), color='black', zorder=1)
            ax.plot((-4, -3), (-4, -5), color='black', zorder=1)
            ax.plot((6, 7), (6, 5), color='black', zorder=1)
            ax.plot((-6, -5), (-6, -7), color='black', zorder=1)
            ax.plot((7, -7), (7, -7), color='black', zorder=1)
            for idx, point in enumerate(input_locations):
                if idx == 0:
                    idx = 1
                ax.plot((point[0] - 1.5 - (idx - 1) * 2, point[0]), (point[1], point[1]), color='black', zorder=1)
                ax.scatter(*point, color=[0, 0.2, 0.6], marker='>', zorder=2, s=2 * self.marker_size)
            flip_appendage = False
            for label, alpha_beta, point in zip(labels, alpha_beta_list, points_list):
                alpha, beta = alpha_beta
                ax.annotate(
                    r"$"+str(beta)+"$", xy=(point[0], point[1] + 0.4), fontsize=self.label_fontsize,
                    horizontalalignment='center', verticalalignment='bottom', color=DARK_BLUE
                )
                ax.annotate(
                    r"$"+str(alpha)+"$", xy=(point[0], point[1] - 0.4), fontsize=self.label_fontsize,
                    horizontalalignment='center', verticalalignment='top', color=DARK_RED
                )
                ax.annotate(
                    label, xy=(point[0] + 0.5, point[1]), fontsize=self.label_fontsize * 1.2,
                    horizontalalignment='right', verticalalignment='center', color=DARK_RED
                )
                ax.scatter(*point, color=DARK_RED, zorder=2, s=2 * self.marker_size)
                add_appendage(point, input=False, upper=flip_appendage)
            ax.plot((0, -6), (0, -6), color='black', zorder=1)
            add_appendage((-6, -6), input=False, upper=True, opaque=True)
        else:
            raise NotImplementedError("Sad")


def plot_node(ax, center, top_ratio, bottom_ratio, top_nullify=False, scale=1):
    top_nullify_color = GRAY if top_nullify else 'black'
    bottom_nullify_color = 'black' if top_nullify else GRAY
    top_nullify_p_color = GRAY if top_nullify else DARK_PURPLE
    bottom_nullify_p_color = DARK_PURPLE if top_nullify else GRAY
    top_nullify_d_color = DARK_PURPLE if top_nullify else LIGHT_PURPLE
    bottom_nullify_d_color = LIGHT_PURPLE if top_nullify else DARK_PURPLE
    bottom_power_color= LIGHT_RED if top_nullify else LIGHT_BLUE
    top_power_color = LIGHT_BLUE if top_nullify else LIGHT_RED
    bottom_power_label_color = DARK_RED if top_nullify else DARK_BLUE
    top_power_label_color = DARK_BLUE if top_nullify else DARK_RED
    bottom_nullify_style = 'x' if top_nullify else 's'
    top_nullify_style = 's' if top_nullify else 'x'
    top_label = r"$\boldsymbol{\alpha}$" if top_nullify else r"$\boldsymbol{\beta}$"
    top_power_label = r"$\boldsymbol{P_\alpha}$" if top_nullify else r"$\boldsymbol{P_\beta}$"
    bottom_label = r"$\boldsymbol{\beta}$" if top_nullify else r"$\boldsymbol{\alpha}$"
    bottom_power_label = r"$\boldsymbol{P_\beta}$" if top_nullify else r"$\boldsymbol{P_\alpha}$"

    ax.scatter(center[0], center[1], color=DARK_RED, zorder=2, s=100 * scale)
    ax.plot((center[0] - scale, center[0]), (center[1] - scale, center[1]), color=top_power_color, linewidth=2.5, zorder=0)
    ax.plot((center[0] - scale, center[0]), (center[1] + scale, center[1]), color=bottom_power_color, linewidth=2.5, zorder=0)
    ax.plot((center[0] + scale, center[0]), (center[1] - scale, center[1]), color=bottom_nullify_p_color, linewidth=2.5, zorder=1)
    ax.plot((center[0] + scale, center[0]), (center[1] + scale, center[1]), color=top_nullify_p_color, linewidth=2.5, zorder=1)
    ax.plot((center[0] + scale, center[0] + 2 * scale),
            (center[1] + scale, center[1] + scale),
            color=top_nullify_p_color, linewidth=2.5, zorder=1)
    ax.plot((center[0] + scale, center[0] + 2 * scale),
            (center[1] - scale, center[1] - scale),
            color=bottom_nullify_p_color, linewidth=2.5, zorder=1)
    ax.scatter(center[0] + 2 * scale, center[1] - scale, color=top_nullify_d_color, marker=top_nullify_style, zorder=2, s=100 * scale)
    ax.scatter(center[0] + 2 * scale, center[1] + scale, color=bottom_nullify_d_color, marker=bottom_nullify_style, zorder=2, s=100 * scale)
    ax.add_patch(Polygon(np.asarray([[-scale, scale], [-(1 + bottom_ratio) * scale, (1 - bottom_ratio) * scale],
                                     [-(1 + bottom_ratio) * scale, (1 + bottom_ratio) * scale]]) + np.asarray(center),
                         color=bottom_power_color))
    ax.add_patch(Polygon(np.asarray([[-scale, -scale], [-(1 + top_ratio) * scale, -(1 + top_ratio) * scale],
                                     [-(1 + top_ratio) * scale, -(1 - top_ratio) * scale]]) + np.asarray(center),
                         color=top_power_color))
    ax.annotate(
        bottom_power_label, xy=(center[0] - 0.5, center[1] - 0.55), fontsize=16,
        horizontalalignment='left', verticalalignment='top', color=top_power_color
    )
    ax.annotate(
        top_power_label, xy=(center[0] - 0.5, center[1] + 0.5), fontsize=20,
        horizontalalignment='left', verticalalignment='bottom', color=bottom_power_color
    )
    ax.annotate(
        bottom_label, xy=(center[0] - (1 + top_ratio / 2 + 0.05) * scale, center[1] - scale), fontsize=16,
        horizontalalignment='center', verticalalignment='center', color=top_power_label_color
    )
    ax.annotate(
        top_label, xy=(center[0] - (1 + bottom_ratio / 2 + 0.1) * scale, center[1] + scale), fontsize=20,
        horizontalalignment='center', verticalalignment='center', color=bottom_power_label_color
    )
    ax.annotate(
        r"$\boldsymbol{T}$", xy=(center[0] - 0.5, center[1]), fontsize=24,
        horizontalalignment='right', verticalalignment='center', color=DARK_RED
    )
    ax.annotate(
        r"$\boldsymbol{P_\alpha} + \boldsymbol{P_\beta}$", xy=(center[0] + 0.5, center[1]), fontsize=24,
        horizontalalignment='left', verticalalignment='center', color=DARK_PURPLE
    )


def plot_node_induction(ax, center, top_nullify=False, scale=2, high_dim=False):
    top_nullify_p_color = GRAY if top_nullify else DARK_PURPLE
    bottom_nullify_p_color = DARK_PURPLE if top_nullify else GRAY
    top_nullify_d_color = DARK_GREEN if top_nullify else LIGHT_PURPLE
    bottom_nullify_d_color = LIGHT_PURPLE if top_nullify else DARK_GREEN
    bottom_power_color = LIGHT_RED if top_nullify else LIGHT_BLUE
    top_power_color = LIGHT_BLUE if top_nullify else LIGHT_RED

    bottom_nullify_style = 'x' if top_nullify else '>'
    top_nullify_style = '>' if top_nullify else 'x'

    dcenter = (center[0] - scale * 2, center[1])
    ax.scatter(dcenter[0], dcenter[1], color=DARK_RED, zorder=2, s=25 * scale)

    ax.plot((center[0], center[0] + scale / 4), (center[1], center[1]),
            color=DARK_PURPLE, linewidth=1.5, zorder=0)
    ax.plot((dcenter[0] - scale / 4, dcenter[0]), (dcenter[1] - scale / 4, dcenter[1]),
            color=top_power_color, linewidth=1, zorder=0)
    ax.plot((dcenter[0] - scale / 4, dcenter[0]), (dcenter[1] + scale / 4, dcenter[1]),
            color=bottom_power_color, linewidth=1, zorder=0)
    ax.plot((dcenter[0] - scale / 2, dcenter[0] - scale / 4), (dcenter[1] - scale / 4, dcenter[1] - scale / 4),
            color=top_power_color, linewidth=1, zorder=0)
    ax.plot((dcenter[0] - scale / 2, dcenter[0] - scale / 4), (dcenter[1] + scale / 4, dcenter[1] + scale / 4),
            color=bottom_power_color, linewidth=1, zorder=0)
    ax.plot((dcenter[0] + scale / 4, dcenter[0]), (dcenter[1] - scale / 4, dcenter[1]),
            color=bottom_nullify_p_color, linewidth=1.5, zorder=0)
    ax.plot((dcenter[0] + scale / 4, dcenter[0]), (dcenter[1] + scale / 4, dcenter[1]),
            color=top_nullify_p_color, linewidth=1.5, zorder=0)
    ax.plot((dcenter[0] + scale / 4, dcenter[0] + scale / 2 + scale / 4 * (1 - top_nullify)),
            (dcenter[1] + scale / 4, dcenter[1] + scale / 4),
            color=top_nullify_p_color, linewidth=1.5, zorder=0)
    ax.plot((dcenter[0] + scale / 4, dcenter[0] + scale / 2 + scale / 4 * top_nullify),
            (dcenter[1] - scale / 4, dcenter[1] - scale / 4),
            color=bottom_nullify_p_color, linewidth=1.5, zorder=0)
    ax.scatter(dcenter[0] + scale / 2 + scale / 4 * (top_nullify), dcenter[1] - scale / 4,
               color=top_nullify_d_color, marker=top_nullify_style,
               zorder=2, s=25 * scale)
    ax.scatter(dcenter[0] + scale / 2 + scale / 4 * (1 - top_nullify), dcenter[1] + scale / 4,
               color=bottom_nullify_d_color, marker=bottom_nullify_style,
               zorder=2, s=25 * scale)

    ax.scatter(dcenter[0] + 3 * scale / 4, dcenter[1] + scale,
               color=DARK_GREEN, marker='>',
               zorder=2, s=25 * scale)
    ax.scatter(dcenter[0] + 3 * scale / 4, dcenter[1] - scale,
               color=DARK_GREEN, marker='>',
               zorder=2, s=25 * scale)
    ax.scatter(dcenter[0] + 3 * scale / 4, dcenter[1] + scale - scale / 4,
               color=DARK_GREEN, marker='>',
               zorder=2, s=25 * scale)
    ax.scatter(dcenter[0] + 3 * scale / 4, dcenter[1] - scale + scale / 4,
               color=DARK_GREEN, marker='>',
               zorder=2, s=25 * scale)
    ax.scatter(dcenter[0] - scale / 2, dcenter[1] - scale / 4,
               color=DARK_GREEN, marker='>',
               zorder=2, s=25 * scale)
    ax.scatter(dcenter[0] - scale / 2, dcenter[1] + scale / 4,
               color=DARK_GREEN, marker='>',
               zorder=2, s=25 * scale)

    ax.plot((dcenter[0] + 3 * scale / 4, dcenter[0] + scale), (dcenter[1] + scale, dcenter[1] + scale),
            color='black', linewidth=1, zorder=0)
    ax.plot((dcenter[0] + 3 * scale / 4, dcenter[0] + scale), (dcenter[1] - scale, dcenter[1] - scale),
            color='black', linewidth=1, zorder=0)
    ax.plot((dcenter[0] + 3 * scale / 4, dcenter[0] + scale), (dcenter[1] + 0.75 * scale, dcenter[1] + 0.75 * scale),
            color='black', linewidth=1, zorder=0)
    ax.plot((dcenter[0] + 3 * scale / 4, dcenter[0] + scale), (dcenter[1] - 0.75 * scale, dcenter[1] - 0.75 * scale),
            color='black', linewidth=1, zorder=0)
    if top_nullify:
        ax.plot((dcenter[0] + 3 * scale / 4, dcenter[0] + scale),
                (dcenter[1] - 0.25 * scale, dcenter[1] - 0.25 * scale),
                color='black', linewidth=1, zorder=0)
    else:
        ax.plot((dcenter[0] + 3 * scale / 4, dcenter[0] + scale),
                (dcenter[1] + 0.25 * scale, dcenter[1] + 0.25 * scale),
                color='black', linewidth=1, zorder=0)

    ax.annotate(
        r"$\boldsymbol{U}_1$", xy=(dcenter[0], dcenter[1] - 0.75), fontsize=12,
        horizontalalignment='center', verticalalignment='center', color=DARK_RED
    )

    ax.annotate(
        r"$\boldsymbol{\vdots}$", xy=(dcenter[0] + 3 * scale / 4, dcenter[1] - 0.5 * scale), fontsize=20,
        horizontalalignment='center', verticalalignment='center', color=DARK_GREEN
    )

    ax.annotate(
        r"$\boldsymbol{\vdots}$", xy=(dcenter[0] + 3 * scale / 4, dcenter[1] + 0.5 * scale), fontsize=20,
        horizontalalignment='center', verticalalignment='center', color=DARK_GREEN
    )

    ax.add_patch(Polygon(np.asarray([[center[0], center[1]], [center[0] - scale, center[1] - scale],
                                     [center[0] - scale, center[1] + scale]]) + np.asarray(center),
                         facecolor=LIGHT_PURPLE, edgecolor=DARK_PURPLE))

    ax.annotate(
        r"$\boldsymbol{R}_{N'}$", xy=(center[0] - scale / 2, center[1]), fontsize=16,
        horizontalalignment='center', verticalalignment='center', color=DARK_PURPLE
    )

