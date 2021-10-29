from simphox.utils import random_unitary
from simphox.circuit import vector_unit
from phox.model.mesh import Mesh
import holoviews as hv
import panel as pn

if __name__ == '__main__':
    hv.extension('bokeh')
    u = random_unitary(8)
    mesh = Mesh(vector_unit(u)[0])
    mesh.v = u.T[-1]
    app = mesh.hvsim(height=200, wide=True)

    pn.serve(app, start=True, show=False, port=5006,
        websocket_origin="localhost:4444",
        ssl_certfile='/home/exx/ssl/jupyter_cert.pem',
        ssl_keyfile='/home/exx/ssl/jupyter_cert.key')
