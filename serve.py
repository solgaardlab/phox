from phox.experiment import TriangularMeshImager
import holoviews as hv
import panel as pn
import pickle

if __name__ == '__main__':
    hv.extension('bokeh')

    with open('configs/amf_config.p', 'rb') as f:
        config = pickle.load(f)

    api = TriangularMeshImager(
        home=(0.05, -0.003),  # mm
        interlayer_xy=(0.0015, -0.3095),  # mm
        spot_xy=(456, 25),
        interspot_xy=(-67, 298),
        config=config,
        stage_port='/dev/ttyUSB3',
        laser_port='/dev/ttyUSB0',
        lmm_port='/dev/ttyUSB2'
    )

    livestream_panel = api.camera.livestream_panel(cmap='gray')
    move_panel = api.stage.move_panel()
    power_panel = api.laser.power_panel()
    wavelength_panel = api.laser.wavelength_panel()
    led_panel = api.led_panel()
    home_panel = api.home_panel()
    propagation_toggle_panel = api.propagation_toggle_panel()
    transparent_panel = api.transparent_button_panel()
    mesh_panel = api.mesh_panel()
    read_panel = api.read_panel()
    calibrate_panel = api.calibrate_panel()

    pn.serve(pn.Column(
        pn.Row(pn.Column(livestream_panel),
           pn.Column(move_panel, home_panel, power_panel,
                     wavelength_panel, led_panel,
                     propagation_toggle_panel, transparent_panel)
          ),
        mesh_panel,
        read_panel,
        calibrate_panel
    ), start=True, show=False, port=5006,
        websocket_origin="localhost:4444",
        ssl_certfile='/home/exx/ssl/jupyter_cert.pem',
        ssl_keyfile='/home/exx/ssl/jupyter_cert.key')
