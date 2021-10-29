from simphox.utils import random_unitary
from simphox.circuit import vector_unit
from phox.model.mesh import Mesh
import holoviews as hv
import panel as pn
import sys

if __name__ == '__main__':
    hv.extension('bokeh')
    u2 = random_unitary(2)
    mesh = Mesh(vector_unit(u2)[0])
    mesh.v = u2.T[-1]
    mzi = mesh.hvsim(height=200, wide=True)
    u4 = random_unitary(4)
    mesh = Mesh(vector_unit(u4)[0])
    mesh.v = u4.T[-1]
    bt4 = mesh.hvsim(height=200, wide=True)
    u8 = random_unitary(8)
    mesh = Mesh(vector_unit(u8)[0])
    mesh.v = u8.T[-1]
    bt8 = mesh.hvsim(height=200, wide=True)
    mesh = Mesh(vector_unit(u8, balanced=False)[0])
    mesh.v = u8.T[-1]
    lc8 = mesh.hvsim(height=200, wide=True)

    pn.serve({
        'mzi': mzi,
        'bt4': bt4,
        'bt8': bt8,
        'lc8': lc8
    }, start=True, show=True, port=int(sys.argv[-1]), websocket_origin='phoxmesh.herokuapp.com')
