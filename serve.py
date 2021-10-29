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

    api = Sputnik(
        home=(0.05, -0.003),  # mm
        interlayer_xy=(0.0015, -0.3095),  # mm
        spot_xy=(456, 25),
        interspot_xy=(-67, 298),
        stage_port='/dev/ttyUSB3',
        laser_port='/dev/ttyUSB0',
        lmm_port='/dev/ttyUSB2',
        ps_calibration=ps_calibration,
    )

    mesh_panel = api.mesh_panel()
    livestream_panel = api.camera.livestream_panel(cmap='gray')
    move_panel = api.stage.move_panel()
    power_panel = api.laser.power_panel()
    spot_panel = api.camera.power_panel()
    wavelength_panel = api.laser.wavelength_panel()
    led_panel = api.led_panel()
    home_panel = api.home_panel()
    propagation_toggle_panel = api.propagation_toggle_panel()
    input_panel = api.input_panel()

    pn.serve(pn.Column(
        pn.Pane(pn.Row(mesh_panel, input_panel), name='Mesh'),
        pn.Row(pn.Tabs(
            ('Live Interface',
             pn.Column(
                 pn.Row(livestream_panel,
                        pn.Column(move_panel, home_panel, power_panel,
                                  wavelength_panel, led_panel,
                                  propagation_toggle_panel)
                        )
             )),
            ('Calibration Panel', api.calibrate_panel())
        ), spot_panel)
    ), start=True, show=False, port=5006,
        websocket_origin="localhost:4444",
        ssl_certfile='/home/exx/ssl/jupyter_cert.pem',
        ssl_keyfile='/home/exx/ssl/jupyter_cert.key')
