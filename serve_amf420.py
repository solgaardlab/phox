from phox.experiment import AMF420Mesh
import holoviews as hv
import panel as pn
import pickle

if __name__ == '__main__':
    hv.extension('bokeh')

    with open('configs/ps_calibration_1560nm_20220313.p', 'rb') as f:
        ps_calibration = pickle.load(f)

    chip = AMF420Mesh(
        home=(0.0, 0.0),  # mm
        interlayer_xy=(0.0015, -0.3095),  # mm
        spot_rowcol=(436, 25),
        interspot_xy=(-67, 298),
        stage_port='/dev/ttyUSB2',
        laser_port='/dev/ttyUSB0',
        ps_calibration=ps_calibration,
        integration_time=1000
    )

    pn.serve(chip.default_panel(), start=True, show=False, port=5006,
        websocket_origin="localhost:4444",
        ssl_certfile='/home/exx/ssl/jupyter_cert.pem',
        ssl_keyfile='/home/exx/ssl/jupyter_cert.key')
