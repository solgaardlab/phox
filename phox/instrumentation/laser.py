import time
from typing import Union, Tuple
import panel as pn

from .serial import SerialMixin


class LaserHP8164A(SerialMixin):
    def __init__(self, port: str = '/dev/ttyUSB2', source_idx: int = 0):
        """Agilent Laser Module with Holoviz GUI interface

        Args:
            port: serial port string
            source_idx: source index on the machine (slot in which the laser is located)
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

    def wavelength_panel(self, wavelength_range: Tuple[float, float] = (1.530, 1.584), dlam: float = 0.001):
        """
        Panel for dispersion handling

        Args:
            wavelength_range: wavelength range
            dlam: lambda step size for adjustment

        Returns:

        """
        dispersion = pn.widgets.FloatSlider(start=wavelength_range[0], end=wavelength_range[1], step=dlam,
                                            value=self.wavelength, name=r'Wavelength (um)',
                                            format='1[.]000')

        def change_wavelength(*events):
            for event in events:
                if event.name == 'value':
                    self.wavelength = event.new

        dispersion.param.watch(change_wavelength, 'value')

        return dispersion

    def power_panel(self, power_range: Tuple[float, float] = (0.05, 4.25), interval=0.001):
        """

        Args:
            power_range: Range of powers
            interval: Interval between the powers that can be specified

        Returns:
            Panel for jupyter notebook for controlling the laser

        """
        power = pn.widgets.FloatSlider(start=power_range[0], end=power_range[1], step=interval,
                                       value=self.power, name='Power (mW)', format='1[.]000')
        state = pn.widgets.Toggle(name='Laser Enable', value=bool(self.state))

        def change_power(*events):
            for event in events:
                if event.name == 'value':
                    self.power = event.new

        def change_state(*events):
            for event in events:
                if event.name == 'value':
                    self.state = int(event.new)

        state.param.watch(change_state, 'value')
        power.param.watch(change_power, 'value')
        return pn.Column(power, state)
