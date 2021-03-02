import time
from typing import Union

from .serial import SerialMixin


class LaserHP8164A(SerialMixin):
    def __init__(self, port: str = '/dev/ttyUSB2', source_idx: int = 0):
        """

        Args:
            port:
            source_idx:
        """
        self.source_idx = source_idx
        SerialMixin.__init__(self,
                             port=port,
                             id_command='*IDN?',
                             id_response='HP8164A',
                             terminator='\r'
                             )
        self.write(f'sour{self.source_idx}:wav?')
        self._wavelength = float(self.read_until('\n')[0]) * 1e6
        self.write(f'sour{self.source_idx}:pow?')
        self._power = float(self.read_until('\n')[0]) * 1000
        self.write(f'sour{self.source_idx}:pow:stat?')
        self._state = int(self.read_until('\n')[0])

    def setup(self):
        self.write('++auto 1')
        self.write('++addr 9')

    @property
    def on(self) -> bool:
        return self.state == 1

    @property
    def state(self) -> int:
        """ Get state of laser (on, off)

        Returns:
            0 (off) or 1 (on)

        """
        return self._state

    @state.setter
    def state(self, state: int):
        """ Set state of laser (on, off)

        Args:
            on:

        """
        self.write(f'sour{self.source_idx}:pow:stat {state}')
        self._state = state

    @property
    def power(self) -> float:
        """ Power of laser in mW

        Returns:
            Laser power in mW
        """
        return self._power

    @power.setter
    def power(self, power: float):
        """Set power of the laser (in mW)

        Args:
            power: laser power in mW

        """
        self.write(f'sour{self.source_idx}:pow {power}mW')
        self._power = power

    @property
    def wavelength(self):
        """Set wavelength of the laser in nm

        Returns:
            Wavelength in um

        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength: float):
        """Set wavelength of the laser in um

        Args:
            wavelength: wavelength in um

        Returns:

        """
        self.write(f'sour{self.source_idx}:wav {wavelength * 1000}NM')
        self._wavelength = wavelength

    def sweep_wavelength(self, start_wavelength: float, stop_wavelength: float, step: float,
                         speed: float, timeout: float):
        """

        Args:
            start_wavelength:
            stop_wavelength:
            step:
            speed:
            timeout:

        Returns:

        """
        self.write(f'wav:swe:star {start_wavelength}nm')
        self.write(f'wav:swe:stop {stop_wavelength}nm')
        self.write(f'wav:swe:step {step}nm')
        self.write(f'wav:swe:spe {speed}nm/s')
        self.write('wav:swe 1')
        time.sleep(timeout)
        self.write('wav:swe 0')
