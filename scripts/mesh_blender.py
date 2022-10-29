from dphox.demo import mesh as device
import bpy
import pickle
import numpy as np

from matplotlib import cm

# turn on bloom
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.eevee.use_bloom = True

all_powers = []
all_ps = []

for material in bpy.data.materials:
    material.user_clear()
    bpy.data.materials.remove(material)

for mesh in bpy.data.meshes:
    mesh.user_clear()
    bpy.data.meshes.remove(mesh)

for collection in bpy.data.collections:
    if collection.name != 'Collection':
        collection.user_clear()
        bpy.data.collections.remove(collection)

with open(bpy.path.abspath('//mesh_image_self_config.p'), 'rb') as f:
    data = pickle.load(f)

    for step in data:
        powers = step['mesh']
        powers = powers / np.sum(powers, axis=0)
        powers = powers[::-1]
        all_powers.append(powers)
        all_ps.append(step['ps'])

power_array, ps_array = device.demo_3d_arrays()


def create_emission_shader(color, strength, mat_name):
    # create a new material resource (with its
    # associated shader)
    mat = bpy.data.materials.new(mat_name)
    # enable the node-graph edition mode
    mat.use_nodes = True

    # clear all starter nodes
    nodes = mat.node_tree.nodes
    nodes.clear()

    # add the Emission node
    node_emission = nodes.new(type="ShaderNodeEmission")
    # (input[0] is the color)
    node_emission.inputs[0].default_value = color
    # (input[1] is the strength)
    node_emission.inputs[1].default_value = strength

    # add the Output node
    node_output = nodes.new(type="ShaderNodeOutputMaterial")

    # link the two nodes
    links = mat.node_tree.links
    link = links.new(node_emission.outputs[0], node_output.inputs[0])

    # return the material reference
    return mat


def power_material(power, name):
    power = 0 if power < 0.05 else power
    color = np.abs(power) * np.array((1, 0.5, 0, 1))
    return create_emission_shader(color, 10, name)


def ps_material(ps, name):
    ps /= np.pi * 2
    color = np.abs(ps) * np.array((0, 1, 0, 1))
    return create_emission_shader(color, 2, name)


for i in range(6):
    for j in range(19):
        power_array[i][j] = power_array[i][j].apply_translation((-device.center[0], -device.center[1], 0))
        ps_array[i][j] = ps_array[i][j].apply_translation((-device.center[0], -device.center[1], 0))

for idx in range(5):
    mesh_collection = bpy.data.collections.new(f'mesh_collection_{idx}')
    bpy.context.scene.collection.children.link(mesh_collection)
    for i in range(6):
        for j in range(19):
            geom = power_array[i][j]
            new_mesh = bpy.data.meshes.new(f'wg_{i}_{j}_{idx}')
            new_mesh.from_pydata(geom.vertices, geom.edges, geom.faces)
            new_mesh.update()
            si = bpy.data.objects.new(f'si_{i}_{j}_{idx}', new_mesh)
            mesh_collection.objects.link(si)
            power_mat = power_material(all_powers[idx][i, j], f'light_{i}_{j}_{idx}')
            si.data.materials.append(power_mat)
            if (j, i) in all_ps[-1]:
                ps_mat = ps_material(all_ps[idx][(j, i)], f'phase_{i}_{j}_{idx}')
                geom = ps_array[-i - 1][j]
                new_mesh = bpy.data.meshes.new(f'ps_{i}_{j}_{idx}')
                new_mesh.from_pydata(geom.vertices, geom.edges, geom.faces)
                new_mesh.update()
                heater = bpy.data.objects.new(f'heater_{i}_{j}_{idx}', new_mesh)
                mesh_collection.objects.link(heater)
                heater.data.materials.append(ps_mat)