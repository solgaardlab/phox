import numpy as np
from descartes import PolygonPatch
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box
from typing import Tuple

from phoxy.components.photonics import MZI
from phoxy.fabrication.material import Material, OXIDE, NITRIDE
from phoxy.components.utils import BACKGROUND_COLOR, PHI_COLOR, THETA_COLOR, GAMMA_COLOR, BEAMSPLITTER_COLOR, HADAMARD_COLOR

DEMO_GAMMA_COLOR = list(np.asarray(GAMMA_COLOR) * 0.75)
DEMO_PHI_COLOR = list(np.asarray(PHI_COLOR) * 0.75)
DEMO_THETA_COLOR = list(np.asarray(THETA_COLOR) * 0.75)
DEMO_PHOTODETECTOR_COLOR = (1, 0, 0, 1)


class SimMZI(MZI):
    def __init__(self, bend_dim: Tuple[float, float], waveguide_width: float,
                 waveguide_thickness: float, phase_shifter_arm_length: float,
                 coupling_spacing: float, interaction_length: float, coupler_end_length: float=0,
                 end_length: float=0, end_bend_dim: Tuple[float, float]=(0, 0),
                 waveguide_material: Material=NITRIDE, substrate_material: Material=OXIDE):
        super(SimMZI, self).__init__(
            bend_dim=bend_dim,
            waveguide_width=waveguide_width,
            waveguide_thickness=waveguide_thickness,
            phase_shifter_arm_length=phase_shifter_arm_length,
            coupling_spacing=coupling_spacing,
            interaction_length=interaction_length,
            coupler_end_length=coupler_end_length,
            end_length=end_length,
            end_bend_dim=end_bend_dim,
            waveguide_material=waveguide_material,
            substrate_material=substrate_material
        )

    def plot_fields(self, ax, fields: np.ndarray, theta: float, phi: float,
                    x_padding_factor: float=1.25, y_padding_factor: float=2.5, label_size=None):
        lower_path, upper_path = self._build_patterns(include_outer_bend=False)
        patches = []
        colors = []
        light_amplitude_mappable = ScalarMappable(cmap='hot')
        light_amplitude_mappable.set_clim(0, 1)
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.set_xlim((-x_padding_factor * self.x_span / 2, x_padding_factor * self.x_span / 2))
        ax.set_ylim((y_padding_factor * self.y_span / 2, -y_padding_factor * self.y_span / 2))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        phase_shifter_thickness = self.waveguide_width * 3
        theta_center = self.end_length + self.end_bend_dim[0] + self.mzi_x_span / 2
        phi_center = theta_center + self.mzi_x_span / 2 - self.coupler_end_length + self.phase_shifter_arm_length / 2
        theta_phase_shifter = box(
            theta_center - self.phase_shifter_arm_length / 2,
            - phase_shifter_thickness / 2,
            theta_center + self.phase_shifter_arm_length / 2,
            + phase_shifter_thickness / 2
        )
        phi_phase_shifter = box(
            phi_center - self.phase_shifter_arm_length / 2,
            - phase_shifter_thickness / 2,
            phi_center + self.phase_shifter_arm_length / 2,
            + phase_shifter_thickness / 2
        )
        if label_size is not None:
            ax.text(theta_center - self.x_span / 2, -self.y_span / 2 - phase_shifter_thickness * 0.75,
                    r'$\boldsymbol{\theta}$', color=THETA_COLOR, horizontalalignment='center', fontsize=label_size)
            ax.text(phi_center - self.x_span / 2, -self.y_span / 2 - phase_shifter_thickness * 0.75,
                    r'$\boldsymbol{\phi}$', color=PHI_COLOR, horizontalalignment='center', fontsize=label_size)
        patches.append(PolygonPatch(
            translate(theta_phase_shifter, -self.x_span / 2, -self.y_span / 2),
            edgecolor='none'))
        colors.append((*THETA_COLOR, theta / np.pi))
        patches.append(PolygonPatch(translate(phi_phase_shifter, -self.x_span / 2, -self.y_span / 2), edgecolor='none'))
        colors.append((*PHI_COLOR, phi / (2 * np.pi)))
        for wvg_idx, wvg_path in enumerate([upper_path, lower_path]):
            for wvg_poly_idx, polygon_point_list in enumerate(reversed(wvg_path.polygons)):
                waveguide_field_patch = PolygonPatch(
                    rotate(Polygon(polygon_point_list), angle=np.pi,
                           origin=(0, 0), use_radians=True), edgecolor='none')
                patches.append(waveguide_field_patch)
                colors.append(
                    light_amplitude_mappable.to_rgba(np.abs(fields[wvg_idx, _mzi_poly_idx_to_layer_num(wvg_poly_idx)]))
                )
        ax.add_collection(PatchCollection(patches, facecolors=colors))

    def plot_demo(self, ax, x_padding_factor: float=1.25, y_padding_factor: float=2.5,
                  label_size=None, label_dist=None, use_hadamard: bool=False):
        ax.set_facecolor('black')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim((-x_padding_factor * self.x_span / 2, x_padding_factor * self.x_span / 2))
        ax.set_ylim((y_padding_factor * self.y_span / 2, -y_padding_factor * self.y_span / 2))
        lower_path, upper_path = self._build_patterns(include_outer_bend=False)
        patches = []
        colors = []
        for wvg_idx, wvg_path in enumerate([upper_path, lower_path]):
            for wvg_poly_idx, polygon_point_list in enumerate(reversed(wvg_path.polygons)):
                waveguide_field_patch = PolygonPatch(
                    rotate(Polygon(polygon_point_list), angle=np.pi,
                           origin=(0, 0), use_radians=True), edgecolor='none')
                patches.append(waveguide_field_patch)
                colors.append(_mzi_poly_idx_to_layer_color(wvg_poly_idx, use_hadamard))
        ax.add_collection(PatchCollection(patches, facecolors=colors))
        if label_size is not None:
            if label_dist is None:
                label_dist = label_size / 2
            for idx in range(2):
                ax.text(-self.x_span / 2 - 1, -self.y_span / 2 + idx * self.y_span, str(idx + 1),
                        horizontalalignment='right', verticalalignment='center', fontsize=label_size, color='yellow')
                ax.text(self.x_span / 2 + 1, -self.y_span / 2 + idx * self.y_span, str(idx + 1),
                        horizontalalignment='left', verticalalignment='center', fontsize=label_size, color='yellow')
            center_x = self.end_length + self.end_bend_dim[0] + self.mzi_x_span / 2 - self.x_span / 2
            bs_distance = self.phase_shifter_arm_length / 2 + self.bend_dim[0] + self.interaction_length / 2
            if use_hadamard:
                ax.text(center_x - bs_distance, -self.y_span / 2 - label_dist, r'$\boldsymbol{H_L}$',
                        horizontalalignment='center', fontsize=label_size, color=HADAMARD_COLOR)
                ax.text(center_x + bs_distance, -self.y_span / 2 - label_dist, r'$\boldsymbol{H_R}$',
                        horizontalalignment='center', fontsize=label_size, color=HADAMARD_COLOR)
            else:
                ax.text(center_x - bs_distance, -self.y_span / 2 - label_dist, r'$\boldsymbol{B_L}$',
                        horizontalalignment='center', fontsize=label_size, color=BEAMSPLITTER_COLOR)
                ax.text(center_x + bs_distance, -self.y_span / 2 - label_dist, r'$\boldsymbol{B_R}$',
                        horizontalalignment='center', fontsize=label_size, color=BEAMSPLITTER_COLOR)
            ax.text(center_x, -self.y_span / 2 - label_dist, r'$\boldsymbol{R_{\theta}}$',
                    horizontalalignment='center', fontsize=label_size, color=THETA_COLOR)
            ax.text(center_x + bs_distance + self.bend_dim[0] + self.coupler_end_length / 2 + self.interaction_length / 2,
                    -self.y_span / 2 - label_dist, r'$\boldsymbol{R_{\phi}}$',
                    horizontalalignment='center', fontsize=label_size, color=PHI_COLOR)

    def plot_control(self, ax, x_padding_factor: float=1.15, y_padding_factor: float=2,
                     label_size=None, basis=None, include_gamma=False):
        lower_path, upper_path = self._build_patterns(include_outer_bend=False)
        patches = []
        colors = []
        light_amplitude_mappable = ScalarMappable(cmap='hot')
        light_amplitude_mappable.set_clim(0, 1)
        # ax.set_facecolor(BACKGROUND_COLOR)
        ax.set_xlim((-x_padding_factor * self.x_span / 2, x_padding_factor * self.x_span / 2))
        ax.set_ylim((y_padding_factor * self.y_span / 2, -y_padding_factor * self.y_span / 2))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        phase_shifter_thickness = self.waveguide_width * 3
        theta_center = self.end_length + self.end_bend_dim[0] + self.mzi_x_span / 2
        phi_center = theta_center + self.mzi_x_span / 2 - self.coupler_end_length + self.phase_shifter_arm_length / 2
        gamma_center = theta_center - self.mzi_x_span / 2 + self.phase_shifter_arm_length / 2
        theta_phase_shifter = box(
            theta_center - self.phase_shifter_arm_length / 2,
            - phase_shifter_thickness / 2,
            theta_center + self.phase_shifter_arm_length / 2 * 0.75,
            + phase_shifter_thickness / 2
        )
        theta_photodetector = box(
            theta_center + self.phase_shifter_arm_length / 2 * 0.75,
            - phase_shifter_thickness / 2,
            theta_center + self.phase_shifter_arm_length / 2,
            + phase_shifter_thickness / 2
        )
        phi_phase_shifter = box(
            phi_center - self.phase_shifter_arm_length / 2,
            - phase_shifter_thickness / 2,
            phi_center + self.phase_shifter_arm_length / 2,
            + phase_shifter_thickness / 2
        )
        phi_photodetector = box(
            phi_center + self.phase_shifter_arm_length / 2 * 0.75,
            - phase_shifter_thickness / 2,
            phi_center + self.phase_shifter_arm_length / 2,
            + phase_shifter_thickness / 2
        )
        gamma_phase_shifter = box(
            gamma_center - self.phase_shifter_arm_length / 2,
            - phase_shifter_thickness / 2,
            gamma_center + self.phase_shifter_arm_length / 2,
            + phase_shifter_thickness / 2
        )
        gamma_photodetector = box(
            gamma_center + self.phase_shifter_arm_length / 2 * 0.75,
            - phase_shifter_thickness / 2,
            gamma_center + self.phase_shifter_arm_length / 2,
            + phase_shifter_thickness / 2
        )
        theta_upper_label = {
            None: r'$\boldsymbol{\theta_{mn}(\theta_{\sigma, mn})}$',
            'sm': r'$\boldsymbol{\theta_{\sigma, mn}}$',
            'hs': r'$\boldsymbol{\theta_{\sigma, mn} / 2}$',
            'sym': r'$\boldsymbol{\theta_{\sigma, mn} / 2}$',
            'asym': r'$\boldsymbol{\theta_{\sigma, mn}}$'
        }
        theta_lower_label = {
            None: r'$\boldsymbol{\theta_{(m+1)n}(\theta_{\sigma, mn})}$',
            'sm': None,
            'hs': r'$\boldsymbol{-\theta_{\sigma, mn} / 2}$',
            'sym': r'$\boldsymbol{-\theta_{\sigma, mn} / 2}$',
            'asym': None
        }
        phi_upper_label = {
            None: r'$\boldsymbol{\phi_{mn}(\phi_{\sigma, mn})}$',
            'sm': r'$\boldsymbol{\phi_{\sigma, mn}}$',
            'hs': r'$\boldsymbol{\phi_{\sigma, mn}}$',
            'sym': r'$\boldsymbol{\phi_{\sigma, mn}}$',
            'asym': r'$\boldsymbol{\phi_{\sigma, mn}}$'
        }
        phi_lower_label = {
            None: r'$\boldsymbol{\phi_{(m+1)n}(\phi_{\sigma, mn})}$',
            'sm': None,
            'hs': None,
            'sym': r'$\boldsymbol{\phi_{\sigma, mn}}$',
            'asym': r'$\boldsymbol{\phi_{\sigma, mn}}$'
        }
        if label_size is not None:
            ax.text(theta_center - self.x_span / 2, -self.y_span / 2 - phase_shifter_thickness * 0.75,
                    theta_upper_label[basis], color=DEMO_THETA_COLOR, horizontalalignment='center', fontsize=label_size)
            ax.text(phi_center - self.x_span / 2, -self.y_span / 2 - phase_shifter_thickness * 0.75,
                    phi_upper_label[basis], color=DEMO_PHI_COLOR, horizontalalignment='center', fontsize=label_size)
            if theta_lower_label[basis] is not None:
                ax.text(theta_center - self.x_span / 2, -self.y_span / 2 + self.mzi_y_span + phase_shifter_thickness * 0.75,
                        theta_lower_label[basis], color=DEMO_THETA_COLOR, verticalalignment='top',
                        horizontalalignment='center', fontsize=label_size)
            if phi_lower_label[basis] is not None:
                ax.text(phi_center - self.x_span / 2, -self.y_span / 2 + self.mzi_y_span + phase_shifter_thickness * 0.75,
                        phi_lower_label[basis], color=DEMO_PHI_COLOR, verticalalignment='top',
                        horizontalalignment='center', fontsize=label_size)
        patches.append(PolygonPatch(
            translate(theta_phase_shifter, -self.x_span / 2, -self.y_span / 2),
            edgecolor='none'))
        colors.append((*DEMO_THETA_COLOR, 1))
        if basis == 'sym' or basis == 'hs' or basis is None:
            patches.append(PolygonPatch(
                translate(theta_phase_shifter, -self.x_span / 2, -self.y_span / 2 + self.mzi_y_span),
                edgecolor='none'))
            colors.append((*DEMO_THETA_COLOR, 1))
        patches.append(PolygonPatch(
            translate(theta_photodetector, -self.x_span / 2, -self.y_span / 2),
            edgecolor='none'))
        colors.append(DEMO_PHOTODETECTOR_COLOR)
        patches.append(PolygonPatch(
            translate(theta_photodetector, -self.x_span / 2, -self.y_span / 2 + self.mzi_y_span),
            edgecolor='none'))
        colors.append(DEMO_PHOTODETECTOR_COLOR)
        patches.append(PolygonPatch(translate(phi_phase_shifter, -self.x_span / 2, -self.y_span / 2), edgecolor='none'))
        colors.append((*DEMO_PHI_COLOR, 1))
        if include_gamma:
            patches.append(PolygonPatch(
                translate(gamma_phase_shifter, -self.x_span / 2, -self.y_span / 2),
                edgecolor='none'))
            colors.append((*DEMO_GAMMA_COLOR, 1))
            patches.append(PolygonPatch(
                translate(gamma_phase_shifter, -self.x_span / 2, -self.y_span / 2 + self.mzi_y_span),
                edgecolor='none'))
            colors.append((*DEMO_GAMMA_COLOR, 1))
            patches.append(PolygonPatch(
                translate(gamma_photodetector, -self.x_span / 2, -self.y_span / 2),
                edgecolor='none'))
            colors.append(DEMO_PHOTODETECTOR_COLOR)
            patches.append(PolygonPatch(
                translate(gamma_photodetector, -self.x_span / 2, -self.y_span / 2 + self.mzi_y_span),
                edgecolor='none'))
            colors.append(DEMO_PHOTODETECTOR_COLOR)

        if basis == 'sym' or basis == 'asym' or basis is None:
            patches.append(PolygonPatch(
                translate(phi_phase_shifter, -self.x_span / 2, -self.y_span / 2 + self.mzi_y_span),
                edgecolor='none'))
            colors.append((*DEMO_PHI_COLOR, 1))
        patches.append(PolygonPatch(
            translate(phi_photodetector, -self.x_span / 2, -self.y_span / 2),
            edgecolor='none'))
        patches.append(PolygonPatch(
            translate(phi_photodetector, -self.x_span / 2, -self.y_span / 2 + self.mzi_y_span),
            edgecolor='none'))
        colors.append(DEMO_PHOTODETECTOR_COLOR)
        colors.append(DEMO_PHOTODETECTOR_COLOR)
        for wvg_idx, wvg_path in enumerate([upper_path, lower_path]):
            for wvg_poly_idx, polygon_point_list in enumerate(reversed(wvg_path.polygons)):
                waveguide_field_patch = PolygonPatch(
                    rotate(Polygon(polygon_point_list), angle=np.pi,
                           origin=(0, 0), use_radians=True), edgecolor='none')
                patches.append(waveguide_field_patch)
                colors.append((0, 0, 0, 1))
        ax.add_collection(PatchCollection(patches, facecolors=colors))

    def plot_pod(self, ax, x_padding_factor: float=1.15, y_padding_factor: float=2, label_size=None):
        lower_path, upper_path = self._build_patterns(include_outer_bend=False)
        patches = []
        colors = []
        ax.set_xlim((-x_padding_factor * self.x_span / 2, x_padding_factor * self.x_span / 2))
        ax.set_ylim((y_padding_factor * self.y_span / 2, -y_padding_factor * self.y_span / 2))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for wvg_idx, wvg_path in enumerate([upper_path, lower_path]):
            for wvg_poly_idx, polygon_point_list in enumerate(reversed(wvg_path.polygons)):
                waveguide_field_patch = PolygonPatch(
                    rotate(Polygon(polygon_point_list), angle=np.pi,
                           origin=(0, 0), use_radians=True), edgecolor='none')
                patches.append(waveguide_field_patch)
                colors.append((0, 0, 0, 1))
        phase_shifter_thickness = self.waveguide_width * 3
        theta_center = self.end_length + self.end_bend_dim[0] + self.mzi_x_span / 2
        phi_center = theta_center + self.mzi_x_span / 2 - self.coupler_end_length + self.phase_shifter_arm_length / 2
        gamma_center = theta_center - self.mzi_x_span / 2 + self.phase_shifter_arm_length / 2
        theta_phase_shifter = box(
            theta_center - self.phase_shifter_arm_length / 2,
            - phase_shifter_thickness / 2,
            theta_center + self.phase_shifter_arm_length / 2,
            + self.mzi_y_span + phase_shifter_thickness / 2
        )
        phi_phase_shifter = box(
            phi_center - self.phase_shifter_arm_length / 2,
            - phase_shifter_thickness / 2,
            phi_center + self.phase_shifter_arm_length / 2,
            + self.mzi_y_span + phase_shifter_thickness / 2
        )
        gamma_phase_shifter = box(
            gamma_center - self.phase_shifter_arm_length / 2,
            - phase_shifter_thickness / 2,
            gamma_center + self.phase_shifter_arm_length / 2,
            + self.mzi_y_span + phase_shifter_thickness / 2
        )
        if label_size is not None:
            ax.text(theta_center - self.x_span / 2, 0, r'$\boldsymbol{R_{\theta}}$',
                    color=DEMO_THETA_COLOR, horizontalalignment='center', fontsize=label_size)
            ax.text(phi_center - self.x_span / 2, 0, r'$\boldsymbol{R_{\phi}}$',
                    color=DEMO_PHI_COLOR, horizontalalignment='center', fontsize=label_size)
            ax.text(gamma_center - self.x_span / 2, 0, r'$\boldsymbol{R_{\gamma}}$',
                    color=DEMO_GAMMA_COLOR, horizontalalignment='center', fontsize=label_size)
            # ax.text(gamma_center - self.x_span / 2, -self.y_span / 2 + phase_shifter_thickness * 0.75,
            #         r'$\boldsymbol{R_{\gamma_1}}$', color=DEMO_GAMMA_COLOR, verticalalignment='top',
            #         horizontalalignment='center', fontsize=label_size)
            # ax.text(gamma_center - self.x_span / 2, -self.y_span / 2 + self.mzi_y_span + phase_shifter_thickness * 0.75,
            #         r'$\boldsymbol{R_{\gamma_2}}$', color=DEMO_GAMMA_COLOR, verticalalignment='top',
            #         horizontalalignment='center', fontsize=label_size)
        patches.append(PolygonPatch(
            translate(theta_phase_shifter, -self.x_span / 2, -self.y_span / 2),
            edgecolor='none'))
        colors.append((*DEMO_THETA_COLOR, 0.5))
        patches.append(PolygonPatch(
            translate(phi_phase_shifter, -self.x_span / 2, -self.y_span / 2),
            edgecolor='none'))
        colors.append((*DEMO_PHI_COLOR, 0.5))
        patches.append(PolygonPatch(
            translate(gamma_phase_shifter, -self.x_span / 2, -self.y_span / 2),
            edgecolor='none'))
        colors.append((*DEMO_GAMMA_COLOR, 0.5))
        # patches.append(PolygonPatch(
        #     translate(gamma_phase_shifter, -self.x_span / 2, -self.y_span / 2 + self.mzi_y_span),
        #     edgecolor='none'))
        # colors.append((*DEMO_GAMMA_COLOR, 0.5))
        ax.add_collection(PatchCollection(patches, facecolors=colors))


def _mzi_poly_idx_to_layer_num(poly_idx):
    if poly_idx <= 2:
        return 0
    elif poly_idx <= 3:
        return 1
    elif poly_idx <= 5:
        return 2
    else:
        return 3


def _mzi_poly_idx_to_layer_color(poly_idx, use_hadamard: bool=False):
    if poly_idx <= 0:
        return 0.5, 0.5, 0.5
    elif poly_idx <= 3:
        return BEAMSPLITTER_COLOR if not use_hadamard else HADAMARD_COLOR
    elif poly_idx <= 4:
        return THETA_COLOR
    elif poly_idx <= 7:
        return BEAMSPLITTER_COLOR if not use_hadamard else HADAMARD_COLOR
    else:
        return PHI_COLOR
