from phox.experiment import Sputnik
import holoviews as hv
import panel as pn
import pickle

if __name__ == '__main__':
    hv.extension('bokeh')

    with open('configs/amf_config.p', 'rb') as f:
        config = pickle.load(f)

    with open('configs/ps_calibration_1557nm_2021_07_18_09_11.p', 'rb') as f:
        ps_calibration = pickle.load(f)

    chip = Sputnik(
        home=(0.05, -0.003),  # mm
        interlayer_xy=(0.0015, -0.3095),  # mm
        spot_xy=(456, 25),
        interspot_xy=(-67, 298),
        stage_port='/dev/ttyUSB2',
        laser_port='/dev/ttyUSB0',
        ps_calibration=ps_calibration,
    )

    pn.serve(chip.default_panel(), start=True, show=False, port=5006,
        websocket_origin="localhost:4444",
        ssl_certfile='/home/exx/ssl/jupyter_cert.pem',
        ssl_keyfile='/home/exx/ssl/jupyter_cert.key')
