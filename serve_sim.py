from simphox.utils import random_unitary
from simphox.circuit import vector_unit, triangular, rectangular, tree_cascade
from phox.model.mesh import Mesh
import holoviews as hv
import panel as pn
import sys

if __name__ == '__main__':
    hv.extension('bokeh')

    def mzi():
        u2 = random_unitary(2)
        mesh = Mesh(vector_unit(u2)[0])
        return mesh.hvsim(height=200, wide=True, title='N=2, MZI')

    def bt4():
        u4 = random_unitary(4)
        mesh = Mesh(vector_unit(u4)[0])
        return mesh.hvsim(height=200, wide=True, title='N=4, Balanced tree')

    def bt8():
        u8 = random_unitary(8)
        mesh = Mesh(vector_unit(u8)[0])
        return mesh.hvsim(height=200, wide=True, title='N=8, Balanced tree')

    def bt16():
        u8 = random_unitary(8)
        mesh = Mesh(vector_unit(u8)[0])
        return mesh.hvsim(height=400, wide=False, title='N=16, Balanced tree')

    def ut8():
        u8 = random_unitary(8)
        mesh = Mesh(vector_unit(u8, balanced=False)[0])
        return mesh.hvsim(height=200, wide=True, title='N=8, Unbalanced tree')

    def tri4():
        u4 = random_unitary(4)
        mesh = Mesh(triangular(u4))
        return mesh.hvsim(height=200, wide=True, title='N=4, Triangular mesh')

    def btc4():
        u4 = random_unitary(4)
        mesh = Mesh(tree_cascade(u4))
        return mesh.hvsim(height=200, wide=True, title='N=4, Tree cascade')

    def btc8():
        u8 = random_unitary(8)
        mesh = Mesh(tree_cascade(u8))
        return mesh.hvsim(height=200, wide=True, title='N=8, Tree cascade')

    def rect4():
        u4 = random_unitary(4)
        mesh = Mesh(rectangular(u4))
        return mesh.hvsim(height=200, wide=True, title='N=4, Rectangular mesh', self_configure_matrix=False)

    pn.serve({
        'MZI N=2': mzi,
        'Balanced Tree N=4': bt4,
        'Balanced Tree N=8': bt8,
        'Balanced Tree N=16': bt16,
        'Unbalanced Tree N=8': ut8,
        'Tree cascade N=4': btc4,
        'Tree cascade N=8': btc8,
        'Triangular N=4': tri4
    }, start=True, show=True, port=int(sys.argv[-1]), websocket_origin='phoxmesh.herokuapp.com')
