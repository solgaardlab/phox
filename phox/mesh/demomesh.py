from typing import Optional, Tuple

from descartes import PolygonPatch
import copy

import numpy as np
from shapely.affinity import translate, rotate
from shapely.geometry import Polygon, box

from neurophox.config import HYPERSPHERICAL, SYMMETRIC, ASYMMETRIC, SINGLEMODE, ORTHOGONAL, GENERAL

from phoxy.components.photonics import MZI, DirectionalCoupler
from phoxy.modules import RMPhotonicModule
from phoxy.components.utils import THETA_COLOR, PHI_COLOR, GAMMA_COLOR

DEMO_PHI_COLOR = list(np.asarray(PHI_COLOR) * 0.75)
DEMO_THETA_COLOR = list(np.asarray(THETA_COLOR) * 0.75)
DEMO_GAMMA_COLOR = list(np.asarray(GAMMA_COLOR) * 0.75)
DEMO_PHOTODETECTOR_COLOR = (1, 0, 0, 1)


class DemoMesh(RMPhotonicModule):
    def __init__(self, mzi: MZI, num_ports: int, end_length: float, depth: int=None, layer: int=0):
        self.mzi = mzi
        self.depth = depth if depth is not None else num_ports
        self.end_bend_dim = (self.mzi.end_bend_dim[0] / 2, self.mzi.end_bend_dim[1] / 2)
        # optional fields for when a simulation needs to be visualized
        self.simulation_patches = []
        self.demo_patches = []
        super(DemoMesh, self).__init__(
            mzi=mzi,
            num_ports=num_ports,
            end_length=end_length,
            layer=layer
        )

    def _add_phase_shifts(self, ax, phase_shifter_thickness: float=None, label_size=None,
                          theta_basis=None, phi_basis=None):
        if not phase_shifter_thickness:
            phase_shifter_thickness = 3 * self.mzi.waveguide_width
        self._add_io_phase_shifters(ax, phase_shifter_thickness, label_size, np.ones((self.num_ports,)))
        self._add_io_phase_shifters(ax, phase_shifter_thickness, label_size, np.ones((self.num_ports,)),
                                    output_shift=self.dim[0] - 2 * self.end_length + self.mzi.phase_shifter_arm_length)
        theta_arrangement = _get_theta_basis_arrangement(self.num_ports, self.num_ports, theta_basis) if theta_basis \
            else np.ones((self.num_ports, self.num_ports))
        valid_mesh_points = np.argwhere(theta_arrangement != 0)
        for point in valid_mesh_points:
            self._add_theta_phase_shifter(ax, phase_shifter_thickness, point, label_size,
                                          ps_type=theta_arrangement[point[0], point[1]])
        phi_arrangement = _get_phi_basis_arrangement(self.num_ports, self.num_ports, phi_basis) if phi_basis \
            else np.zeros((self.num_ports, self.num_ports))
        valid_mesh_points = np.argwhere(phi_arrangement != 0)
        for point in valid_mesh_points:
            self._add_phi_phase_shifter(ax, phase_shifter_thickness, point, label_size,
                                        ps_type=phi_arrangement[point[0], point[1]])

    def plot_basis(self, ax, phi_basis: str):
        if phi_basis == SINGLEMODE or phi_basis == HYPERSPHERICAL:
            phi_arrangement = backward_cmf_arrangements(self.num_ports, self.num_ports)[-1]
        else:
            phi_arrangement = downward_cmf_arrangements(self.num_ports, self.num_ports)[-1]
        self.plot_mesh(ax, 1.25, 1.5)
        self._add_io_phase_shifters(ax, None, None, phi_arrangement[0][:, 0])
        self._cmf_demo_panel(ax, phi_arrangement[0][:, 1:], phi_arrangement[1])

    def _cmf_demo_panel(self, ax, phi_arrangement_l: np.ndarray, phi_arrangement_r: np.ndarray,
                        phase_shifter_thickness: float=None, label_size=None, theta_basis=SINGLEMODE):
        if not phase_shifter_thickness:
            phase_shifter_thickness = 3 * self.mzi.waveguide_width
        theta_arrangement = _get_theta_basis_arrangement(self.num_ports, self.num_ports, theta_basis)
        valid_mesh_points = np.argwhere(theta_arrangement != 0)
        for point in valid_mesh_points:
            self._add_theta_phase_shifter(ax, phase_shifter_thickness, point, label_size)

        valid_mesh_points = np.argwhere(phi_arrangement_l != 0)
        for point in valid_mesh_points:
            self._add_phi_phase_shifter(ax, phase_shifter_thickness, point, label_size,
                                        ps_type=phi_arrangement_l[point[0], point[1]],
                                        ps_shift=True)

        valid_mesh_points = np.argwhere(phi_arrangement_r != 0)
        for point in valid_mesh_points:
            self._add_phi_phase_shifter(ax, phase_shifter_thickness, point, label_size,
                                        ps_type=phi_arrangement_r[point[0], point[1]])

    def plot_cmf_demo(self, plt, phase_shifter_thickness: float=None, phi_basis: str=SINGLEMODE,
                  x_padding_factor=1.25, y_padding_factor=1.25,
                  label_size=None, theta_basis=SINGLEMODE, dpi=200):
        if phi_basis == SINGLEMODE or phi_basis == HYPERSPHERICAL:
            phi_arrangements = backward_cmf_arrangements(self.num_ports, self.num_ports)
            title = r'Horizontal Common Mode Flow: $N = ' + str(self.num_ports) + '$ Mesh'
        else:
            phi_arrangements = downward_cmf_arrangements(self.num_ports, self.num_ports)
            title = r'Vertical Common Mode Flow: $N = ' + str(self.num_ports) + '$ Mesh'
        fig, axes = plt.subplots(len(phi_arrangements), 1, figsize=(10, 10), dpi=dpi)
        for idx, phi_arrangement in enumerate(phi_arrangements):
            self.plot_mesh(axes[idx], x_padding_factor, y_padding_factor)
            self._add_io_phase_shifters(axes[idx], phase_shifter_thickness, label_size,
                                        phi_arrangement[0][:, 0])
            self._cmf_demo_panel(axes[idx], phi_arrangement[0][:, 1:], phi_arrangement[1],
                                 phase_shifter_thickness, label_size, theta_basis)
            axes[idx].text(-0.05, 0.5, f'{idx + 1}', transform=axes[idx].transAxes, size=18, verticalalignment='center')
        plt.suptitle(title, fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _construct_mesh(self, ax):
        for wvg_idx, wvg_path in enumerate(self.fill_pattern):
            self.num_poly_layers = len(wvg_path.polygons)
            for wvg_poly_idx, polygon_point_list in enumerate(reversed(wvg_path.polygons)):
                waveguide_field_patch = PolygonPatch(
                    rotate(translate(Polygon(polygon_point_list), -self.dim[0] / 2, -self.dim[1] / 2),
                           angle=np.pi, origin=(0, 0), use_radians=True),
                    edgecolor='none', color=(0, 0, 0))
                ax.add_patch(waveguide_field_patch)

    def plot_mesh(self, ax, x_padding_factor=1.25, y_padding_factor=1.25):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim((-x_padding_factor * self.dim[0] / 2, x_padding_factor * self.dim[0] / 2))
        ax.set_ylim((y_padding_factor * self.dim[1] / 2, -y_padding_factor * self.dim[1] / 2))
        self._construct_mesh(ax)

    def plot_compact_demo(self, ax, x_padding_factor=1.25, y_padding_factor=1.5):
        self.plot_mesh(ax, x_padding_factor, y_padding_factor)
        self._add_phase_shifts(ax)

    def _add_theta_phase_shifter(self, ax, phase_shifter_thickness: float,
                                  point: Tuple[float, float], label_size=None,
                                 ps_type=2, is_compact=False):
        is_compact = self.mzi.end_bend_dim[0] < 1
        center_x = point[1] * (self.mzi.mzi_x_span + self.mzi.x_span) / 2 + self.mzi.mzi_x_span / 2
        center_y = point[0] * self.interport_distance
        center_x += self.end_length + self.end_bend_dim[0] + self.mzi.bend_dim[0] / 2 * is_compact\
                    + self.mzi.bend_dim[0] / 2 - self.mzi.phase_shifter_arm_length / 2
        center_y += self.mzi.end_bend_dim[1] / 2
        if ps_type == 1:
            theta_phase_shifter = box(
                center_x - self.mzi.phase_shifter_arm_length / 2,
                center_y - phase_shifter_thickness / 2,
                center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
                center_y + phase_shifter_thickness / 2
            )
            theta_photodetector = box(
                center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
                center_y - phase_shifter_thickness / 2,
                center_x + self.mzi.phase_shifter_arm_length / 2,
                center_y + phase_shifter_thickness / 2
            )
            ax.add_patch(PolygonPatch(translate(theta_phase_shifter, -self.dim[0] / 2, -self.dim[1] / 2),
                                                   edgecolor='none', color=DEMO_THETA_COLOR, alpha=0.5))
            ax.add_patch(PolygonPatch(
                translate(theta_photodetector, -self.dim[0] / 2, -self.dim[1] / 2),
                edgecolor='none', color=DEMO_THETA_COLOR))
        else:
            top_ps = box(
                center_x - self.mzi.phase_shifter_arm_length / 2,
                center_y - phase_shifter_thickness / 2,
                center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
                center_y + phase_shifter_thickness / 2
            )
            bottom_ps = box(
                center_x - self.mzi.phase_shifter_arm_length / 2,
                center_y + self.mzi.mzi_y_span - phase_shifter_thickness / 2,
                center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
                center_y + self.mzi.mzi_y_span + phase_shifter_thickness / 2
            )
            cm_ps = box(
                center_x - self.mzi.phase_shifter_arm_length / 2,
                center_y - phase_shifter_thickness / 2,
                center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
                center_y + self.mzi.mzi_y_span + phase_shifter_thickness / 2
            )
            cm_pd = box(
                center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
                center_y - phase_shifter_thickness / 2,
                center_x + self.mzi.phase_shifter_arm_length / 2,
                center_y + self.mzi.mzi_y_span + phase_shifter_thickness / 2
            )
            ax.add_patch(PolygonPatch(
                translate(top_ps, -self.dim[0] / 2, -self.dim[1] / 2), color=DEMO_THETA_COLOR, alpha=0.5))
            ax.add_patch(PolygonPatch(
                translate(bottom_ps, -self.dim[0] / 2, -self.dim[1] / 2), color=DEMO_THETA_COLOR, alpha=0.5))
            ax.add_patch(PolygonPatch(
                translate(cm_ps, -self.dim[0] / 2, -self.dim[1] / 2), edgecolor=DEMO_THETA_COLOR, facecolor='none'))
            ax.add_patch(PolygonPatch(
                translate(cm_pd, -self.dim[0] / 2, -self.dim[1] / 2), color=DEMO_THETA_COLOR))
        if label_size is not None:
            ax.text(center_x - self.dim[0] / 2, center_y - self.dim[1] / 2 - phase_shifter_thickness * 0.75,
                    r'$\boldsymbol{\theta_{' + str(point[0] + 1) + str(point[1] + 1) + '}}$', color=DEMO_THETA_COLOR,
                    horizontalalignment='center', fontsize=label_size)

    def _add_phi_phase_shifter(self, ax, phase_shifter_thickness: float,
                               point: Tuple[float, float], label_size=None,
                               ps_type=1, ps_shift: bool=False, is_compact: bool=False):
        is_compact = self.mzi.end_bend_dim[0] < 1
        phi_color = DEMO_PHI_COLOR if ps_shift else DEMO_GAMMA_COLOR
        center_x = point[1] * (self.mzi.mzi_x_span + self.mzi.x_span) / 2 + self.mzi.mzi_x_span
        center_y = point[0] * self.interport_distance
        center_x += self.end_length + self.end_bend_dim[0] + self.mzi.bend_dim[0] * 1.5 * is_compact - self.mzi.bend_dim[0] / 2 - \
                    self.mzi.phase_shifter_arm_length / 2 * (self.num_ports % 2)
        center_y += self.mzi.end_bend_dim[1] / 2
        phi_phase_shifter = box(
            center_x - self.mzi.phase_shifter_arm_length / 2 + ps_shift * 1.25 * self.mzi.phase_shifter_arm_length,
            center_y - phase_shifter_thickness / 2,
            center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75 + ps_shift * 1.25 * self.mzi.phase_shifter_arm_length,
            center_y + phase_shifter_thickness / 2 + (ps_type - 1) * self.mzi.mzi_y_span
        )
        phi_photodetector = box(
            center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75 + ps_shift * 1.25 * self.mzi.phase_shifter_arm_length,
            center_y - phase_shifter_thickness / 2,
            center_x + self.mzi.phase_shifter_arm_length / 2 + ps_shift * 1.25 * self.mzi.phase_shifter_arm_length,
            center_y + phase_shifter_thickness / 2 + (ps_type - 1) * self.mzi.mzi_y_span
        )
        if label_size is not None:
            ax.text(center_x - self.dim[0] / 2, center_y - self.dim[1] / 2 - phase_shifter_thickness * 0.75,
                    r'$\boldsymbol{\phi_{' + str(point[0] + 1) + str(point[1] + 1) + '}}$', color=phi_color,
                    horizontalalignment='center', fontsize=label_size)
        phi_phase_shift_patch = PolygonPatch(
            translate(phi_phase_shifter, -self.dim[0] / 2, -self.dim[1] / 2),
            edgecolor='none', color=phi_color, alpha=0.5)
        phi_photodetector_patch = PolygonPatch(
            translate(phi_photodetector, -self.dim[0] / 2, -self.dim[1] / 2),
            edgecolor='none', color=phi_color)
        ax.add_patch(phi_phase_shift_patch)
        ax.add_patch(phi_photodetector_patch)

    def _add_io_phase_shifters(self, ax, phase_shifter_thickness, label_size,
                               mask: Optional[np.ndarray]=None, output_shift: float=0):
        if not phase_shifter_thickness:
            phase_shifter_thickness = 3 * self.mzi.waveguide_width
        for idx in range(self.num_ports):
            if mask is not None and mask[idx] == 1:
                x = self.end_length + output_shift
                center_y = idx * self.interport_distance
                io_phase_shifter = box(
                    x - self.mzi.phase_shifter_arm_length,
                    center_y - phase_shifter_thickness / 2,
                    x - self.mzi.phase_shifter_arm_length / 2 * 0.25,
                    center_y + phase_shifter_thickness / 2
                )
                io_photodetector = box(
                    x - self.mzi.phase_shifter_arm_length / 2 * 0.25,
                    center_y - phase_shifter_thickness / 2,
                    x,
                    center_y + phase_shifter_thickness / 2
                )
                io_phase_shift_patch = PolygonPatch(
                    translate(io_phase_shifter, -self.dim[0] / 2, -self.dim[1] / 2),
                    edgecolor='none', color=DEMO_PHI_COLOR, alpha=0.5
                )
                io_photodetector_patch = PolygonPatch(
                    translate(io_photodetector, -self.dim[0] / 2, -self.dim[1] / 2),
                    edgecolor='none', color=DEMO_PHI_COLOR)
                if label_size is not None:
                    ax.text(self.end_length - self.dim[0] / 2,
                            center_y - self.dim[1] / 2 - phase_shifter_thickness * 0.75,
                            r'$\boldsymbol{\phi_{0' + str(idx + 1) + '}}$', color=DEMO_PHI_COLOR,
                            horizontalalignment='center', fontsize=label_size)
                ax.add_patch(io_phase_shift_patch)
                ax.add_patch(io_photodetector_patch)

    def component_demo(self, plt, phase_shifter_thickness: Optional[float]=None, dpi: int=200):
        phase_shifter_thickness = 3 * self.waveguide_width if not phase_shifter_thickness else phase_shifter_thickness
        center_x = self.mzi.phase_shifter_arm_length
        lower_path = box(0, -self.waveguide_width / 2, 2 * self.mzi.phase_shifter_arm_length, self.waveguide_width / 2)
        upper_path = box(0, self.mzi.mzi_y_span - self.waveguide_width / 2,
                         2 * self.mzi.phase_shifter_arm_length,
                         self.waveguide_width / 2 + self.mzi.mzi_y_span)

        dc = DirectionalCoupler(**self.mzi.dc_config_properties)
        dc.coupler_end_length = self.mzi.phase_shifter_arm_length - self.mzi.bend_dim[0]

        top_ps = box(
            center_x - self.mzi.phase_shifter_arm_length / 2,
            -phase_shifter_thickness / 2,
            center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
            phase_shifter_thickness / 2
        )
        top_pd = box(
            center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
            -phase_shifter_thickness / 2,
            center_x + self.mzi.phase_shifter_arm_length / 2,
            phase_shifter_thickness / 2
        )
        bottom_ps = box(
            center_x - self.mzi.phase_shifter_arm_length / 2,
            self.mzi.mzi_y_span - phase_shifter_thickness / 2,
            center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
            self.mzi.mzi_y_span + phase_shifter_thickness / 2
        )
        bottom_pd = box(
            center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
            self.mzi.mzi_y_span - phase_shifter_thickness / 2,
            center_x + self.mzi.phase_shifter_arm_length / 2,
            self.mzi.mzi_y_span + phase_shifter_thickness / 2
        )
        cm_ps = box(
            center_x - self.mzi.phase_shifter_arm_length / 2,
            -phase_shifter_thickness / 2,
            center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
            self.mzi.mzi_y_span + phase_shifter_thickness / 2
        )
        cm_pd = box(
            center_x + self.mzi.phase_shifter_arm_length / 2 * 0.75,
            -phase_shifter_thickness / 2,
            center_x + self.mzi.phase_shifter_arm_length / 2,
            self.mzi.mzi_y_span + phase_shifter_thickness / 2
        )

        _, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=dpi)

        axes_titles = [
            r"$B = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & i\\ i & 1 \end{bmatrix}$",
            r"$W(\theta_1, \theta_2) = \begin{bmatrix} e^{i \theta_1} & 0\\ 0 & e^{i \theta_2} \end{bmatrix}$",
            r"$L(\theta) = \begin{bmatrix} e^{i \theta} & 0\\ 0 & 1 \end{bmatrix}$",
            r"$R(\theta) = \begin{bmatrix} 1 & 0\\ 0 & e^{i \theta} \end{bmatrix}$",
            r"$C(\theta) =\begin{bmatrix} e^{i \theta} & 0\\ 0 & e^{i \theta} \end{bmatrix}$",
            r"$D(\theta) = \begin{bmatrix} e^{i \theta / 2} & 0\\ 0 & e^{-i \theta / 2} \end{bmatrix}$"
        ]

        for idx, ax in enumerate(axes.T.flatten()):
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_xlim((- self.mzi.phase_shifter_arm_length / 2, 2.5 * self.mzi.phase_shifter_arm_length))
            ax.set_ylim((- self.mzi.mzi_y_span / 2, 1.5 * self.mzi.mzi_y_span))
            if idx > 0:
                ax.add_patch(PolygonPatch(upper_path, color='black'))
                ax.add_patch(PolygonPatch(lower_path, color='black'))
            else:
                dc.plot(ax, facecolor='black')
            if idx == 1 or idx == 3 or idx == 5:
                ax.add_patch(PolygonPatch(top_ps, color=DEMO_PHOTODETECTOR_COLOR, alpha=0.5))
                ax.add_patch(PolygonPatch(top_pd, color=DEMO_PHOTODETECTOR_COLOR))
            if idx == 1 or idx == 2 or idx == 5:
                ax.add_patch(PolygonPatch(bottom_ps, color=DEMO_PHOTODETECTOR_COLOR, alpha=0.5))
                ax.add_patch(PolygonPatch(bottom_pd, color=DEMO_PHOTODETECTOR_COLOR))
            if idx == 4:
                ax.add_patch(PolygonPatch(cm_ps, color=DEMO_PHOTODETECTOR_COLOR, alpha=0.5))
                ax.add_patch(PolygonPatch(cm_pd, color=DEMO_PHOTODETECTOR_COLOR))
            if idx == 5:
                ax.add_patch(PolygonPatch(cm_ps, facecolor='none', edgecolor=DEMO_PHOTODETECTOR_COLOR))
                ax.add_patch(PolygonPatch(cm_pd, color=DEMO_PHOTODETECTOR_COLOR))
            ax.set_title(axes_titles[idx], fontsize=20)
            ax.axis('off')
        plt.tight_layout()

    def cmf_update_demo(self, plt, phase_shifter_thickness: Optional[float]=None, dpi: int=200):
        phase_shifter_thickness = 3 * self.waveguide_width if not phase_shifter_thickness else phase_shifter_thickness

        lower_path = box(self.mzi.mzi_x_span / 2, - self.mzi.mzi_y_span / 2 - self.waveguide_width / 2,
                         self.mzi.mzi_x_span / 2 + 2 * self.mzi.phase_shifter_arm_length,
                         -self.mzi.mzi_y_span / 2 + self.waveguide_width / 2)
        upper_path = box(self.mzi.mzi_x_span / 2, self.mzi.mzi_y_span / 2 - self.waveguide_width / 2,
                         self.mzi.mzi_x_span / 2 + 2 * self.mzi.phase_shifter_arm_length,
                         self.waveguide_width / 2 + self.mzi.mzi_y_span / 2)

        top_ps = box(
            -self.mzi.phase_shifter_arm_length / 2,
            - self.mzi.mzi_y_span / 2 - phase_shifter_thickness / 2,
            self.mzi.phase_shifter_arm_length / 2 * 0.75,
            - self.mzi.mzi_y_span / 2 + phase_shifter_thickness / 2
        )
        top_pd = box(
            self.mzi.phase_shifter_arm_length / 2 * 0.75,
            -self.mzi.mzi_y_span / 2 - phase_shifter_thickness / 2,
            self.mzi.phase_shifter_arm_length / 2,
            -self.mzi.mzi_y_span / 2 + phase_shifter_thickness / 2
        )
        bottom_ps = box(
            -self.mzi.phase_shifter_arm_length / 2,
            self.mzi.mzi_y_span / 2 - phase_shifter_thickness / 2,
            self.mzi.phase_shifter_arm_length / 2 * 0.75,
            self.mzi.mzi_y_span / 2 + phase_shifter_thickness / 2
        )
        bottom_pd = box(
            self.mzi.phase_shifter_arm_length / 2 * 0.75,
            self.mzi.mzi_y_span / 2 - phase_shifter_thickness / 2,
            self.mzi.phase_shifter_arm_length / 2,
            self.mzi.mzi_y_span / 2 + phase_shifter_thickness / 2
        )
        cm_ps = box(
            -self.mzi.phase_shifter_arm_length / 2,
            -self.mzi.mzi_y_span / 2 - phase_shifter_thickness / 2,
            self.mzi.phase_shifter_arm_length / 2 * 0.75,
            self.mzi.mzi_y_span / 2 + phase_shifter_thickness / 2
        )
        cm_pd = box(
            self.mzi.phase_shifter_arm_length / 2 * 0.75,
            - self.mzi.mzi_y_span / 2 - phase_shifter_thickness / 2,
            self.mzi.phase_shifter_arm_length / 2,
            self.mzi.mzi_y_span / 2 + phase_shifter_thickness / 2
        )

        _, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=dpi)

        axes_titles = [
            "Overparametrized MZI",
            "Horizontal CMF Update",
            "Vertical CMF Update",
        ]

        for idx, ax in enumerate(axes.T.flatten()):
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_xlim((- self.mzi.mzi_x_span / 2 * 1.25, self.mzi.mzi_x_span / 2 * 1.25 + 2 * self.mzi.phase_shifter_arm_length))
            ax.set_ylim((-self.mzi.mzi_y_span, self.mzi.mzi_y_span))
            ax.add_patch(PolygonPatch(upper_path, facecolor='black', edgecolor='none'))
            ax.add_patch(PolygonPatch(lower_path, facecolor='black', edgecolor='none'))
            self.mzi.plot(ax, facecolor='black')
            ax.add_patch(PolygonPatch(bottom_ps, color=DEMO_THETA_COLOR, alpha=0.5))
            ax.add_patch(PolygonPatch(top_ps, color=DEMO_THETA_COLOR, alpha=0.5))
            ax.add_patch(PolygonPatch(cm_ps, edgecolor=DEMO_THETA_COLOR, facecolor='none'))
            ax.add_patch(PolygonPatch(cm_pd, color=DEMO_THETA_COLOR))
            ax.add_patch(
                PolygonPatch(translate(top_ps, -self.mzi.bend_dim[0] * 2 - self.mzi.phase_shifter_arm_length),
                             color=DEMO_PHI_COLOR, alpha=0.5))
            ax.add_patch(
                PolygonPatch(translate(top_pd, -self.mzi.bend_dim[0] * 2 - self.mzi.phase_shifter_arm_length),
                             color=DEMO_PHI_COLOR))
            if idx == 0:
                ax.add_patch(
                    PolygonPatch(translate(bottom_ps, self.mzi.bend_dim[0] * 2 + 2.25 * self.mzi.phase_shifter_arm_length),
                                 color=DEMO_PHI_COLOR, alpha=0.5))
                ax.add_patch(
                    PolygonPatch(translate(bottom_pd, self.mzi.bend_dim[0] * 2 + 2.25 * self.mzi.phase_shifter_arm_length),
                                 color=DEMO_PHI_COLOR))
            if idx == 0 or idx == 2:
                ax.add_patch(
                    PolygonPatch(
                        translate(top_ps, self.mzi.bend_dim[0] * 2 + 2.25 * self.mzi.phase_shifter_arm_length),
                        color=DEMO_PHI_COLOR, alpha=0.5))
                ax.add_patch(
                    PolygonPatch(
                        translate(top_pd, self.mzi.bend_dim[0] * 2 + 2.25 * self.mzi.phase_shifter_arm_length),
                        color=DEMO_PHI_COLOR))
            if idx == 0 or idx == 1:
                ax.add_patch(
                    PolygonPatch(translate(bottom_ps, -self.mzi.bend_dim[0] * 2 - self.mzi.phase_shifter_arm_length),
                                 color=DEMO_PHI_COLOR, alpha=0.5))
                ax.add_patch(
                    PolygonPatch(translate(bottom_pd, -self.mzi.bend_dim[0] * 2 - self.mzi.phase_shifter_arm_length),
                                 color=DEMO_PHI_COLOR))
            if idx == 1:
                ax.add_patch(
                    PolygonPatch(translate(bottom_ps, self.mzi.bend_dim[0] * 2 + self.mzi.phase_shifter_arm_length),
                                 color=DEMO_GAMMA_COLOR, alpha=0.5))
                ax.add_patch(
                    PolygonPatch(translate(bottom_pd, self.mzi.bend_dim[0] * 2 + self.mzi.phase_shifter_arm_length),
                                 color=DEMO_GAMMA_COLOR))
            if idx == 2:
                ax.add_patch(
                    PolygonPatch(translate(cm_ps, self.mzi.bend_dim[0] * 2 + self.mzi.phase_shifter_arm_length),
                                 color=DEMO_GAMMA_COLOR, alpha=0.5))
                ax.add_patch(
                    PolygonPatch(translate(cm_pd, self.mzi.bend_dim[0] * 2 + self.mzi.phase_shifter_arm_length),
                                 color=DEMO_GAMMA_COLOR))
            ax.set_title(axes_titles[idx], fontsize=20)
            ax.axis('off')
        plt.tight_layout()


def _get_checkerboard(dim: int, num_layers: int):
    checkerboard = np.zeros((dim - 1, num_layers))
    checkerboard[::2, ::2] = checkerboard[1::2, 1::2] = 1
    return checkerboard


def _get_theta_basis_arrangement(dim: int, num_layers: int, basis: str):
    checkerboard = _get_checkerboard(dim, num_layers)
    cases = {
        GENERAL: np.ones((dim, num_layers)),
        HYPERSPHERICAL: checkerboard * 3,
        ORTHOGONAL: checkerboard * 3,
        SYMMETRIC: checkerboard * 3,
        SINGLEMODE: checkerboard,
        ASYMMETRIC: checkerboard
    }
    return cases[basis]


def _get_phi_basis_arrangement(dim: int, num_layers: int, basis: str):
    checkerboard = _get_checkerboard(dim, num_layers)
    cases = {
        GENERAL: np.ones((dim, num_layers)),
        HYPERSPHERICAL: checkerboard,
        ORTHOGONAL: np.zeros((dim, num_layers + 1)),
        SYMMETRIC: checkerboard * 2,
        SINGLEMODE: checkerboard,
        ASYMMETRIC: checkerboard * 2
    }
    return cases[basis]


def backward_cmf_arrangements(dim: int, num_layers: int):
    phi_arrangements = []

    gamma = np.random.rand(dim)
    external_phase_shifts = np.random.rand(dim, num_layers)

    phase_shifts = np.hstack((gamma[:, np.newaxis], external_phase_shifts)).T
    single_mode_layers = np.zeros_like(external_phase_shifts.T)

    phi_arrangements.append((copy.deepcopy(phase_shifts.T != 0) * 1, copy.deepcopy(single_mode_layers.T != 0) * 1))

    for i in range(num_layers):
        current_layer = num_layers - i
        start_idx = (current_layer - 1) % 2
        end_idx = dim - (current_layer + dim - 1) % 2
        # calculate phase information
        upper_phase = phase_shifts[current_layer][start_idx:end_idx][::2]
        lower_phase = phase_shifts[current_layer][start_idx:end_idx][1::2]
        # assign differential phase to the single mode layer and keep common mode layer
        single_mode_layers[-i - 1][start_idx:end_idx][::2] = upper_phase - lower_phase
        phase_shifts[current_layer][start_idx:end_idx][::2] = lower_phase
        # update the previous layer with the common mode calculated for the current layer
        phase_shifts[current_layer - 1] += phase_shifts[current_layer]
        phase_shifts[current_layer] = 0
        phi_arrangements.append((copy.deepcopy(phase_shifts.T != 0) * 1, copy.deepcopy(single_mode_layers.T != 0) * 1))

    return phi_arrangements


def downward_cmf_arrangements(dim: int, num_layers: int):
    phi_arrangements = []

    gamma = np.random.rand(dim)
    external_phase_shifts = np.random.rand(dim, num_layers)

    phase_shifts = np.hstack((gamma[:, np.newaxis], external_phase_shifts))
    common_mode_layers = np.zeros_like(external_phase_shifts)

    phi_arrangements.append((copy.deepcopy(phase_shifts != 0) * 1, copy.deepcopy(common_mode_layers != 0) * 2))

    if not num_layers % 2:  # annoying edge case
        phase_shifts[0, -2] += phase_shifts[0, -1]
        phase_shifts[0, -1] = 0

    # step 1: run common mode flow downward
    for idx in range(dim - 1):
        start_idx = idx % 2
        end_idx = num_layers + (idx + num_layers) % 2
        left_phase = phase_shifts[idx][start_idx:end_idx][::2]
        right_phase = phase_shifts[idx][start_idx:end_idx][1::2]
        common_mode_layers[idx][start_idx:end_idx][::2] = left_phase + right_phase
        phase_shifts[idx + 1][start_idx:end_idx] -= phase_shifts[idx][start_idx:end_idx]
        phase_shifts[idx][start_idx:end_idx] = 0
        phi_arrangements.append((copy.deepcopy(phase_shifts != 0) * 1, copy.deepcopy(common_mode_layers != 0) * 2))

    if dim % 2:
        phase_shifts[-1, 1:] = 0
    else:
        end_idx = num_layers + 1 - num_layers % 2
        phase_shifts[-1][1:end_idx][::2] = 0
        common_mode_layers[-2:] = 0
    phi_arrangements.append((copy.deepcopy(phase_shifts != 0) * 1, copy.deepcopy(common_mode_layers != 0) * 2))

    return phi_arrangements
