import time
from typing import Union
import numpy as np

from .serial import SerialMixin


class LightwaveMultimeterHP8163A(SerialMixin):
    def __init__(self, port: str = '/dev/ttyUSB2', source_idx: int = 2, gpib_addr: int = 18):
        self.source_idx = source_idx
        self.gpib_addr = gpib_addr
        SerialMixin.__init__(self,
                             port=port,
                             id_command='*IDN?',
                             id_response='HP8163A',
                             terminator='\r'
                             )

    def setup(self):
        self.write('++auto 1')
        self.write(f'++addr {self.gpib_addr}')

    def power(self, meas_time: float = 0.5) -> float:
        """Get power of laser (in mW)

        Args:
            meas_time: Amount of time to measure powers

        Returns:
            Laser power in mW

        """

        self.write(f'read{self.source_idx}:pow?')
        time.sleep(meas_time)
        return float(self.read_until('\n')[0])

        # self.write(f'sens{self.source_idx}:func:stat logg,star')
        # time.sleep(meas_time)
        # self.write(f'sens{self.source_idx}:func:res?')
        # self._ser.read()
        # n = int(self._ser.read())
        # self._ser.read(n)
        # b = self._ser.read_until(b'\n')
        # return np.frombuffer(b[:-1], dtype=np.float32)

