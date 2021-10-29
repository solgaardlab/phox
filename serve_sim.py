from simphox.utils import random_unitary
from simphox.circuit import vector_unit
from phox.model.mesh import Mesh
import holoviews as hv
import panel as pn
import sys

if __name__ == '__main__':
    hv.extension('bokeh')
    u = random_unitary(8)
    mesh = Mesh(vector_unit(u)[0])
    mesh.v = u.T[-1]
    app = mesh.hvsim(height=200, wide=True)

    pn.serve(app, start=True, show=True, port=int(sys.argv[-1]), websocket_origin='phoxmesh.herokuapp.com')
