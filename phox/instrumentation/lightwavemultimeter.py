import time
from typing import Union

from .serial import SerialMixin


class LightwaveMultimeterHP8163A(SerialMixin):
    def __init__(self, port: str = '/dev/ttyUSB2', source_idx: int = 1):
        self.source_idx = source_idx
        SerialMixin.__init__(self,
                             port=port,
                             id_command='*IDN?',
                             id_response='HP8163A',
                             terminator='\r'
                             )

    def setup(self):
        self.write('++auto 1')
        self.write('++addr 9')

    @property
    def power(self) -> float:
        """ Get state of laser (on, off)

        Returns:
            0 (off) or 1 (on)

        """
        self.write(f'read{self.source_idx}:pow?')
        return float(self.read_until('\n')[0]) * 1000
