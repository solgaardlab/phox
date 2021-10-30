from simphox.utils import random_unitary
from simphox.circuit import vector_unit
from phox.model.mesh import Mesh
import holoviews as hv
import panel as pn
import sys

if __name__ == '__main__':
    hv.extension('bokeh')

    def get_page_user():
        u2 = random_unitary(2)
        mesh = Mesh(vector_unit(u2)[0])
        mesh.v = u2.T[-1]
        mzi = mesh.hvsim(height=200, wide=True, title='MZI')
        u4 = random_unitary(4)
        mesh = Mesh(vector_unit(u4)[0])
        mesh.v = u4.T[-1]
        bt4 = mesh.hvsim(height=200, wide=True, title='Binary tree with N=4')
        u8 = random_unitary(8)
        mesh = Mesh(vector_unit(u8)[0])
        mesh.v = u8.T[-1]
        bt8 = mesh.hvsim(height=200, wide=True, title='Binary tree with N=8')
        mesh = Mesh(vector_unit(u8, balanced=False)[0])
        mesh.v = u8.T[-1]
        lc8 = mesh.hvsim(height=200, wide=True, title='Linear change with N=8')
        return {
            'MZI': mzi,
            'BalancedTree4': bt4,
            'BalancedTree8': bt8,
            'LinearChain8': lc8
        }

    pn.serve(get_page_user, start=True, show=True, port=int(sys.argv[-1]), websocket_origin='phoxmesh.herokuapp.com')
