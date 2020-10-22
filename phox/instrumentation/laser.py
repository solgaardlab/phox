"""
HEWLETT-PACKARD, 8164A (Tunable Laser Source)
We use GPIB-USB Interface to communicate
"""

import visa
import time


class LaserHP8164A:
    def __init__(self, port_number: int = 0, source_idx: int = 0):
        rm = visa.ResourceManager()
        address = rm.list_resources()[port_number]
        self.source_idx = source_idx
        self.instrument = rm.open_resource(address)
        self.instrument.read_termination = '\n'
        self.instrument.write_termination = '\r\n'
        # TODO(sunil): figure out how to do this verification
        self.instrument.query('*IDN?')

    def set_power(self, power: float):
        self.instrument.write(f'sour{self.source_idx}:pow {power}mW')

    def set_wavelength(self, wavelength: float):
        self.instrument.write(f'sour{self.source_idx}:wav {wavelength}NM')

    def sweep_wavelength(self, start_wavelength: float, stop_wavelength: float, step: float,
                         speed: float, timeout: float):
        self.instrument.write(f'wav:swe:star {start_wavelength}nm')
        self.instrument.write(f'wav:swe:stop {stop_wavelength}nm')
        self.instrument.write(f'wav:swe:step {step}nm')
        self.instrument.write(f'wav:swe:spe {speed}nm/s')
        self.instrument.write('wav:swe 1')
        time.sleep(timeout)
        self.instrument.write('wav:swe 0')

# power_set(my_instrument, 3)
# wavelength_sweep(my_instrument, 1530, 1560, 10, 10, 10)
# wavelength_sweep(my_instrument, 1550)
