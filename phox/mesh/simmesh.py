from typing import Callable, Optional

from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
import matplotlib.animation as manimation

import numpy as np
from shapely.affinity import translate, rotate
from shapely.geometry import Polygon, box

from neurophox.fields import ERDFields
from neurophox.phases import RDPhases

from phoxy.components.photonics import MZI
from phoxy.modules import RMPhotonicModule
from phoxy.components.utils import BEAMSPLITTER_COLOR, THETA_COLOR, \
    PHI_COLOR, GAMMA_COLOR, MAX_THETA, MAX_PHI, BACKGROUND_COLOR, HADAMARD_COLOR


class SimMesh(RMPhotonicModule):
    def __init__(self, mzi: MZI, num_ports: int, end_length: float, depth: int=None,
                 layer: int=0, use_dmzi: bool=False):
        self.mzi = mzi
        self.depth = depth if depth is not None else num_ports
        self.end_bend_dim = (self.mzi.end_bend_dim[0] / 2, self.mzi.end_bend_dim[1] / 2)
        self.use_dmzi = use_dmzi
        # optional fields for when a simulation needs to be visualized
        self.plotted_simulated_fields = False
        self.simulation_patches = []
        self.demo_patches = []
        self.simulation_patches_rgba = []
        self.demo_patches_rgba = []
        self.valid_mesh_points = np.asarray(_get_rectangular_mesh_points(num_ports, self.depth)).T
        super(SimMesh, self).__init__(
            mzi=mzi,
            num_ports=num_ports,
            end_length=end_length,
            layer=layer
        )

    def _add_simulated_phases(self, ax, phase_shifter_thickness: float, phases: RDPhases, label_size=None,
                              layer_range=None):
        if phases is not None:
            for idx in range(self.num_ports):
                if layer_range and (layer_range[0] > 0):
                    continue
                center_x = self.end_length - self.mzi.phase_shifter_arm_length / 2 - \
                           self.mzi.phase_shifter_arm_length / 2 * (self.num_ports % 2)
                center_y = idx * self.interport_distance
                gamma_phase_shifter = box(
                    center_x - self.mzi.phase_shifter_arm_length / 2,
                    center_y - phase_shifter_thickness / 2,
                    center_x + self.mzi.phase_shifter_arm_length / 2,
                    center_y + phase_shifter_thickness / 2
                )
                gamma_phase_shift_patch = PolygonPatch(
                    translate(gamma_phase_shifter, -self.dim[0] / 2, -self.dim[1] / 2),
                    edgecolor='none'
                )
                if label_size is not None:
                    ax.text(center_x - self.dim[0] / 2, center_y - self.dim[1] / 2 - phase_shifter_thickness * 0.75,
                            '$\\gamma_{'+str(idx + 1)+'}$', color=GAMMA_COLOR, horizontalalignment='center',
                            fontsize=label_size)
                self.simulation_patches.append(gamma_phase_shift_patch)
                # print(phases.gamma[idx])
                self.simulation_patches_rgba.append((*GAMMA_COLOR, 1))
                # self.simulation_patches_rgba.append((*GAMMA_COLOR, np.mod(phases.gamma[idx], 2* np.pi) / MAX_PHI))
            # if phases.basis == ASYMMETRIC or phases.basis == SINGLEMODE:
            #     theta_arrangement = phases.theta.checkerboard
            # else:
            #     theta_arrangement = phases.theta.differential_mode_arrangement
            theta_arrangement = phases.theta.checkerboard
            valid_mesh_points = np.argwhere(theta_arrangement != 0)
            for point in valid_mesh_points:
                if layer_range and (point[1] < layer_range[0] or point[1] > layer_range[1]):
                    continue
                center_x = point[1] * (self.mzi.mzi_x_span + self.mzi.x_span) / 2
                center_y = point[0] * self.interport_distance
                center_x += self.end_length + self.end_bend_dim[0] + self.mzi.mzi_x_span / 2 - self.mzi.phase_shifter_arm_length / 2 * (self.num_ports % 2)
                center_y += self.mzi.end_bend_dim[1] / 2
                theta_phase_shifter = box(
                    center_x - self.mzi.phase_shifter_arm_length / 2,
                    center_y - phase_shifter_thickness / 2,
                    center_x + self.mzi.phase_shifter_arm_length / 2,
                    center_y + phase_shifter_thickness / 2
                )
                theta_phase_shift_patch = PolygonPatch(translate(theta_phase_shifter, -self.dim[0] / 2, -self.dim[1] / 2),
                                                       edgecolor='none')
                if label_size is not None:
                    ax.text(center_x - self.dim[0] / 2, center_y - self.dim[1] / 2 - phase_shifter_thickness * 0.75,
                            '$\\theta_{'+str(point[0] + 1)+str(point[1] + 1)+'}$', color=THETA_COLOR,
                            horizontalalignment='center', fontsize=label_size)
                self.simulation_patches.append(theta_phase_shift_patch)
                # self.simulation_patches_rgba.append((*THETA_COLOR, np.mod(np.abs(theta_arrangement[point[0], point[1]]), 2 * np.pi) / MAX_THETA))
                self.simulation_patches_rgba.append((*THETA_COLOR, 1))
            # if phases.basis == HYPERSPHERICAL or phases.basis == SINGLEMODE:
            #     setpoint_arrangement = phases.setpoint.checkerboard
            #     setpoint_varname = 'phi'
            # else:
            #     setpoint_arrangement = phases.setpoint.push_pull_arrangement
            #     setpoint_varname = 'zeta'
            phi_arrangement = phases.phi.checkerboard
            valid_mesh_points = _get_rectangular_mesh_points(dim=phi_arrangement.shape[0], num_layers=phi_arrangement.shape[1])
            for point in valid_mesh_points:
                if layer_range and (point[1] < layer_range[0] or point[1] > layer_range[1]):
                    continue
                center_x = point[1] * (self.mzi.mzi_x_span + self.mzi.x_span) / 2
                center_y = point[0] * self.interport_distance
                center_x += self.end_length + self.end_bend_dim[0] + self.mzi.mzi_x_span - \
                           self.mzi.phase_shifter_arm_length / 2 * (self.num_ports % 2)
                center_y += self.mzi.end_bend_dim[1] / 2
                phi_phase_shifter = box(
                    center_x - self.mzi.phase_shifter_arm_length / 2,
                    center_y - phase_shifter_thickness / 2,
                    center_x + self.mzi.phase_shifter_arm_length / 2,
                    center_y + phase_shifter_thickness / 2
                )
                if label_size is not None:
                    ax.text(center_x - self.dim[0] / 2, center_y - self.dim[1] / 2 - phase_shifter_thickness * 0.75,
                            f'$\phi'+'_{'+str(point[0] + 1)+str(point[1] + 1)+'}$', color=PHI_COLOR,
                            horizontalalignment='center', fontsize=label_size)
                phi_phase_shift_patch = PolygonPatch(translate(phi_phase_shifter, -self.dim[0] / 2, -self.dim[1] / 2),
                                                     edgecolor='none')
                self.simulation_patches.append(phi_phase_shift_patch)
                # self.simulation_patches_rgba.append((*PHI_COLOR, np.mod(phi_arrangement[point[0], point[1]], 2 * np.pi) / MAX_PHI))
                self.simulation_patches_rgba.append((*PHI_COLOR, 1))

    def _construct_mesh_from_fields(self, fields: np.ndarray, layer_range=None):
        self.light_amplitude_mappable = ScalarMappable(cmap='gray')
        self.light_amplitude_mappable.set_clim(0, 1)
        for wvg_idx, wvg_path in enumerate(self.fill_pattern):
            self.num_poly_layers = len(wvg_path.polygons)
            for wvg_poly_idx, polygon_point_list in enumerate(reversed(wvg_path.polygons)):
                layer = _mzi_poly_idx_to_layer_num(wvg_poly_idx)
                if layer > fields.shape[0] - 1:
                    layer = fields.shape[0] - 1
                if layer_range and (layer // 4 < layer_range[0] or layer // 4 > layer_range[1]):
                    continue
                waveguide_field_patch = PolygonPatch(
                    rotate(translate(Polygon(polygon_point_list), -self.dim[0] / 2, -self.dim[1] / 2),
                                      angle=np.pi, origin=(0, 0), use_radians=True),
                    edgecolor='none')
                self.simulation_patches.append(waveguide_field_patch)
                self.simulation_patches_rgba.append(
                    self.light_amplitude_mappable.to_rgba(np.sqrt(np.abs(fields[layer, self.num_ports - 1 - wvg_idx])))
                )

    def plot_field_propagation(self, ax, fields: ERDFields, phases: RDPhases, layer_range=None,
                               phase_shifter_thickness=None, x_padding_factor=1.25, y_padding_factor=1.25,
                               replot=False, label_size=None):

        if not self.plotted_simulated_fields or replot:
            self.simulation_patches = []
            self.simulation_patches_rgba = []
            ax.set_facecolor(list(np.asarray(BEAMSPLITTER_COLOR) * 0.75))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_xlim((-x_padding_factor * self.dim[0] / 2, x_padding_factor * self.dim[0] / 2))
            ax.set_ylim((y_padding_factor * self.dim[1] / 2, -y_padding_factor * self.dim[1] / 2))
            if not phase_shifter_thickness:
                phase_shifter_thickness = 3 * self.mzi.waveguide_width
            self._add_simulated_phases(ax, phase_shifter_thickness, phases, label_size, layer_range=layer_range)
            self._construct_mesh_from_fields(fields.fields, layer_range=layer_range)
            self.simulation_patch_collection = PatchCollection(
                self.simulation_patches,
                facecolors=self.simulation_patches_rgba
            )
            ax.add_collection(self.simulation_patch_collection)
            self.plotted_simulated_fields = True
        else:
            self.simulation_patches_rgba = []
            for idx in range(self.num_ports):
                self.simulation_patches_rgba.append((*GAMMA_COLOR, phase_shift_layer[idx] / MAX_PHI))
            for point in self.valid_mesh_points:
                self.simulation_patches_rgba.append((*THETA_COLOR, theta_checkerboard[point[0], point[1]] / MAX_THETA))
            for point in self.valid_mesh_points:
                self.simulation_patches_rgba.append((*PHI_COLOR, phi_checkerboard[point[0], point[1]] / MAX_PHI))
            for wvg_idx in range(self.num_ports):
                for wvg_poly_idx in range(self.num_poly_layers):
                    layer = _mzi_poly_idx_to_layer_num(wvg_poly_idx)
                    if layer > fields.shape[1] - 1:
                        layer = fields.shape[1] - 1
                    self.simulation_patches_rgba.append(
                        self.light_amplitude_mappable.to_rgba(np.abs(fields[self.num_ports - 1 - wvg_idx, layer]))
                    )
            self.simulation_patch_collection.set_facecolors(self.simulation_patches_rgba)

    def animate_field_propagation(self, plt, title: str, save_path: str, movie_name: str, fields: np.ndarray,
                                  theta_checkerboard: np.ndarray=None, phi_checkerboard: np.ndarray=None,
                                  phase_shift_layer: np.ndarray=None, input_to_propagate: int=None,
                                  phase_shifter_thickness: float=None, x_padding_factor: float=1.25,
                                  y_padding_factor: float=1.25, pbar_handle: Callable=None,
                                  movie_fileext: str='mp4', dpi: int=500):
        fig = plt.figure(dpi=dpi)
        fig.clear()
        ax = plt.axes()
        ax.set_title(title)
        input_to_propagate = input_to_propagate if input_to_propagate is not None else self.num_ports // 2

        def plot_to_layer(layer):
            self.plot_field_propagation(
                ax,
                np.hstack([fields[:layer, :, input_to_propagate].T,
                           np.zeros((fields.shape[2], fields.shape[0] - layer - 1))]),
                theta_checkerboard,
                phi_checkerboard,
                phase_shift_layer,
                phase_shifter_thickness=phase_shifter_thickness,
                x_padding_factor=x_padding_factor,
                y_padding_factor=y_padding_factor,
                replot=(layer == 0)
            )
            fig.canvas.draw()

        filetype_dict = {
            "gif": "imagemagick",
            "mp4": "ffmpeg"
        }
        if not (movie_fileext in filetype_dict):
            raise Exception('Must specify either mp4 or gif (mp4 highly recommended).')
        ffmpeg_writer = manimation.writers[filetype_dict[movie_fileext]]
        writer = ffmpeg_writer(fps=10,
                               metadata=dict(
                                   title=f"{movie_name}",
                                   artist="Matplotlib")
                               )

        layer_iterator = pbar_handle(range(fields.shape[0])) if pbar_handle else range(fields.shape[0])

        with writer.saving(fig, f"{save_path}/{movie_name}.{movie_fileext}", dpi=dpi):
            for layer in layer_iterator:
                plot_to_layer(layer)
                writer.grab_frame()


# Assume each simulation MZI layer has 10 segments (polygons)
def _mzi_poly_idx_to_layer_num(poly_idx):
    if poly_idx == 0:
        return 0
    else:
        layer_num_start = (poly_idx - 4) // 10 * 4 + 1
        poly_idx_offset = (poly_idx - 4) % 10
        if poly_idx_offset <= 2:
            return layer_num_start
        elif poly_idx_offset <= 3:
            return layer_num_start + 1
        elif poly_idx_offset <= 6:
            return layer_num_start + 2
        else:
            return layer_num_start + 3


# Assume each simulation MZI layer has 10 segments (polygons)
def _mzi_poly_idx_to_layer_color(poly_idx):
    if poly_idx < 3:
        return GAMMA_COLOR
    else:
        poly_idx_offset = (poly_idx - 4) % 10
        if poly_idx_offset <= 1:
            return BEAMSPLITTER_COLOR
        elif poly_idx_offset <= 2:
            return THETA_COLOR
        elif poly_idx_offset <= 5:
            return BEAMSPLITTER_COLOR
        elif poly_idx_offset <= 8:
            return PHI_COLOR
        else:
            return BEAMSPLITTER_COLOR


def _get_rectangular_mesh_points(dim, num_layers):
    checkerboard = np.zeros((dim - 1, num_layers))
    checkerboard[::2, ::2] = checkerboard[1::2, 1::2] = 1
    return np.where(checkerboard == 1)
