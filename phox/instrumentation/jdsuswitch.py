from .serial import SerialMixin


class JDSUSwitch(SerialMixin):
    def __init__(self, port: str = '/dev/ttyUSB3'):
        """Agilent Laser Module with Holoviz GUI interface

        Args:
            port: serial port string
            source_idx: source index on the machine (slot in which the laser is located)
        """
        SerialMixin.__init__(self,
                             port=port,
                             id_command='IDN?',
                             id_response='141137',  # serial number
                             terminator='\r',
                             baudrate=2400
                             )
        # self.enable_rs485()
        # self.write(f'sour{self.source_idx}:wav?')
        # self._wavelength = float(self.read_until('\n')[0]) * 1e6
        # self.write(f'sour{self.source_idx}:pow?')
        # self._power = float(self.read_until('\n')[0]) * 1000
        # self.write(f'sour{self.source_idx}:pow:stat?')
        # self._state = int(self.read_until('\n')[0])
