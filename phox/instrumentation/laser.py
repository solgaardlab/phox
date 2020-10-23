"""
HEWLETT-PACKARD, 8164A (Tunable Laser Source)
We use GPIB-USB Interface to communicate
"""
import time
from .base import SerialMixin


class LaserHP8164A(SerialMixin):
    def __init__(self, port: str = '/dev/ttyUSB4', source_idx: int = 0):
        self.source_idx = source_idx
        SerialMixin.__init__(self,
                             port=port,
                             id_command='*IDN?',
                             id_response='HP8164A',
                             terminator='\r'
                             )

    def setup(self):
        self.write('++auto 1')
        self.write('++addr 9')

    def set_power(self, power: float):
        self.write(f'sour{self.source_idx}:pow {power}mW')

    def set_wavelength(self, wavelength: float):
        self.write(f'sour{self.source_idx}:wav {wavelength}NM')

    def sweep_wavelength(self, start_wavelength: float, stop_wavelength: float, step: float,
                         speed: float, timeout: float):
        self.write(f'wav:swe:star {start_wavelength}nm')
        self.write(f'wav:swe:stop {stop_wavelength}nm')
        self.write(f'wav:swe:step {step}nm')
        self.write(f'wav:swe:spe {speed}nm/s')
        self.write('wav:swe 1')
        time.sleep(timeout)
        self.write('wav:swe 0')

# power_set(my_instrument, 3)
# wavelength_sweep(my_instrument, 1530, 1560, 10, 10, 10)
# wavelength_sweep(my_instrument, 1550)
