from simphox.utils import random_unitary, random_vector
from simphox.circuit import triangular, rectangular, tree_cascade, unbalanced_tree, balanced_tree
from phox.model.mesh import Mesh
import holoviews as hv
import panel as pn
import sys

if __name__ == '__main__':
    hv.extension('bokeh')

    def mzi():
        mesh = Mesh(balanced_tree, random_unitary(2))
        return mesh.hvsim(height=200, wide=True, title='N=2, MZI', self_configure_matrix=False)

    def bt4():
        mesh = Mesh(balanced_tree, random_unitary(4))
        return mesh.hvsim(height=200, wide=True, title='N=4, Balanced tree', self_configure_matrix=False)

    def bt8():
        mesh = Mesh(balanced_tree, random_unitary(8))
        return mesh.hvsim(height=200, wide=True, title='N=8, Balanced tree', self_configure_matrix=False)

    def bt16():
        mesh = Mesh(balanced_tree, random_unitary(16))
        return mesh.hvsim(height=400, wide=False, title='N=16, Balanced tree', self_configure_matrix=False)

    def ut4():
        mesh = Mesh(unbalanced_tree, random_unitary(4))
        return mesh.hvsim(height=200, wide=True, title='N=4, Unbalanced tree', self_configure_matrix=False)

    def ut8():
        mesh = Mesh(unbalanced_tree, random_unitary(8))
        return mesh.hvsim(height=200, wide=True, title='N=8, Unbalanced tree', self_configure_matrix=False)

    def tri4():
        mesh = Mesh(triangular, random_unitary(4))
        return mesh.hvsim(height=200, wide=True, title='N=4, Triangular mesh')

    def tri8():
        mesh = Mesh(triangular, random_unitary(8))
        return mesh.hvsim(height=200, wide=True, title='N=4, Triangular mesh')

    def btc4():
        mesh = Mesh(tree_cascade, random_unitary(4))
        return mesh.hvsim(height=200, wide=True, title='N=4, Tree cascade')

    def btc8():
        mesh = Mesh(tree_cascade, random_unitary(8))
        return mesh.hvsim(height=200, wide=True, title='N=8, Tree cascade')

    def rect4():
        mesh = Mesh(rectangular, random_unitary(4))
        return mesh.hvsim(height=200, wide=True, title='N=4, Rectangular mesh', self_configure_matrix=False)

    def rect8():
        mesh = Mesh(rectangular, random_unitary(8))
        return mesh.hvsim(height=200, wide=True, title='N=8, Rectangular mesh', self_configure_matrix=False)

    pn.serve({
        'MZI2': mzi,
        'BalancedTree4': bt4,
        'BalancedTree8': bt8,
        'BalancedTree16': bt16,
        'UnbalancedTree4': ut4,
        'UnbalancedTree8': ut8,
        'TreeCascade4': btc4,
        'TreeCascade8': btc8,
        'Triangular4': tri4,
        'Triangular8': tri8,
        'Rectangular4': rect4,
        'Rectangular8': rect8,
    }, start=True, show=True,
        port=int(sys.argv[-1]), websocket_origin='phoxmesh.herokuapp.com')
