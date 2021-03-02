__author__ = 'Joe Landry'

from .serial import SerialMixin
import logging as logger
import abc
import numpy as np
import os
import ctypes

ERROR_CODES = {
    -1: 'Unknown Command',
    -2: 'Unrecognized Axis Parameter',
    -3: 'Missing Parameters',
    -4: 'Parameter Out of Range',
    -5: 'Operation Failed',
    -6: 'Undefined Error D:',
    -21: 'Serial Command Halted by HALT',
}

fpr = r'[-+]?(?:[0-9]*[.])?[0-9]+'


class Axis(object):
    X = 0
    Y = 1


class Stage(object):

    @abc.abstractmethod
    def status(self, axis): raise NotImplementedError

    @abc.abstractmethod
    def move(self, **kwargs): raise NotImplementedError

    @abc.abstractmethod
    def move_rel(self, **kwargs): raise NotImplementedError


class ASI(SerialMixin, Stage):
    MAX_SPEED = 7.5  # [mm/s]

    """
    Object responsible for managing ASI Imaging stage.
    """

    def __init__(self, port='/dev/ttyUSB2', x_limits=(-10, 10), y_limits=(-20, 20)):
        SerialMixin.__init__(self, port)
        self.stage_config = {
            'X Limits': x_limits,
            'Y Limits': y_limits,
            'Zero Button': 0,
            'Fast Axis': Axis.X
        }
        self.x_limits = x_limits
        self.y_limits = y_limits
        self._is_on = False

    def status(self, axis):
        return Command(send_expr=f'I {axis}', read_expr='Maintain code', timeout=1.0).execute(self)

    def move(self, x=None, y=None):
        """
        Move to a specific stage location
        :param x: x position [mm]
        :param y: y position [mm]
        :return:
        """
        args = ['' if x is None else f'X={self.mm_to_encoder(x)}',
                '' if y is None else f'Y={self.mm_to_encoder(y)}']
        Command(send_expr='MOVE {0} {1}'.format(*args), read_expr=':A', timeout=1).execute(self)

    def move_rel(self, x=None, y=None):

        args = ['' if x is None else f'X={self.mm_to_encoder(x)}',
                '' if y is None else f'Y={self.mm_to_encoder(y)}']
        Command(send_expr='R {0} {1}'.format(*args), read_expr=':A', timeout=1).execute(self)

    def reset(self):
        Command(send_expr='~', read_expr='RESET:', timeout=5.0).execute(self)

    def where(self):
        read_expr = f':A\s({fpr})\s({fpr})\s\\r\\n'
        ret = Command(send_expr='WHERE X Y', read_expr=read_expr, group=(1, 2)).execute(self)
        return tuple(self.encoder_to_mm(float(_)) for _ in ret)

    def halt(self):
        Command(send_expr='HALT', read_expr=r'(:A|:N-21)').execute(self)

    def speed(self):
        read_expr = f':A\sX=({fpr})\s+Y=({fpr})\s\\r\\n'
        ret = Command(send_expr='S X? Y?', read_expr=read_expr, group=(1, 2)).execute(self)
        return tuple((float(_)) for _ in ret)

    def set_speed(self, x, y):
        """
        Set stage speed
        :param x: x-axis speed [mm/s]
        :param y: y-axis speed [mm/s]
        :return:
        """
        args = ['' if x is None else 'X={0}'.format(np.clip(x, 0.0, self.MAX_SPEED)),
                '' if y is None else 'Y={0}'.format(np.clip(y, 0.0, self.MAX_SPEED))]
        Command(send_expr='S {0} {1}'.format(*args), read_expr=':A').execute(self)

    def who(self):
        return Command(send_expr='WHO', read_expr=r':A\s(\S+)\s+\r\n', group=1).execute(self)

    def info(self, y: bool = False):
        return Command(send_expr='INFO Y' if y else 'INFO X', read_expr=f':A\s{fpr}\\r\\n', group=1).execute(self)

    def version(self):
        return Command(send_expr='VERSION', read_expr=r':A Version:\s(\S+)\s', group=1).execute(self)

    def is_moving(self):
        return Command(send_expr='/', read_expr=r'(\w)\r\n', group=1).execute(self) == 'B'

    def set_home(self):
        raise NotImplementedError('Needs to be defined')

    def home(self):
        read_expr = f':A\sX=({fpr})\s+Y=({fpr})\s\\r\\n'
        ret = Command(send_expr='HM X? Y?', read_expr=read_expr, group=(1, 2)).execute(self)
        return tuple((float(_)) for _ in ret)

    def zero(self):
        Command(send_expr='Z', read_expr=':A').execute(self)

    def kp(self, val: float, y: bool = True):
        return Command(send_expr=f'KP Y={val}' if y else f'KP X={val}', read_expr=':A').execute(self)

    def set_limits(self, x_lim=None, y_lim=None):
        low_args, high_args = [], []

        def format_lim_args(lim, axis):
            if (len(lim) != 2) | (lim[0] > lim[1]):
                raise ValueError(f'{axis} Limit not formatted properly')
            low_args.append(f'{axis}={lim[0]}')
            high_args.append(f'{axis}={lim[1]}')

        if x_lim is not None:
            format_lim_args(x_lim, 'X')
        if y_lim is not None:
            format_lim_args(y_lim, 'Y')

        Command(send_expr='SL {0} {1}'.format(*low_args), read_expr=':A').execute(self)
        Command(send_expr='SU {0} {1}'.format(*high_args), read_expr=':A').execute(self)

    def setup(self):
        self.set_button_enable(zero=self.stage_config['Zero Button'])
        self.set_limits(x_lim=self.stage_config['X Limits'], y_lim=self.stage_config['Y Limits'])

    # --- Scanning ---
    def setup_scan(self, x_lim, y_lim, num_lines=1, serpentine=False):
        # X, Y, Z = 0 define unused axes. F=0 means raster, 1 means serpentine
        Command(send_expr='SN X=1 Y=2 Z=0 F={0}'.format(int(serpentine)), read_expr=':A').execute(
            self)  # TODO: Change fast axis using config file
        Command(send_expr='NR X={0} Y={1}'.format(x_lim[0], x_lim[1])).execute(self)
        Command(send_expr='NV X={0} Y={1} Z={2} F=1.0'.format(y_lim[0], y_lim[1], num_lines)).execute(self)
        Command(send_expr='TTL X=1', read_expr=':A').execute(self)  # Unsure if necessary

    def start_scan(self):
        Command(send_expr='SN', read_expr=':A').execute(self)

    def close(self):
        """
        Turn instrument off when shutting down
        :return: None
        """
        pass

    zero_button = 0
    home_button = 1
    at_button = 2
    joystick_button = 3

    def set_button_enable(self, zero=1, home=1, at=1, joystick=1):
        # Bit 0: "Zero" Button
        # Bit 1: "Home" Button
        # Bit 2: "@" Button
        # Bit 3: Joystick Button
        Command(send_expr=f'BE Z=1111{zero}{home}{at}{joystick}', read_expr=':A').execute(self)

    def mm_to_encoder(self, mm):
        return round(mm * 1e4)

    def encoder_to_mm(self, enc):
        return enc * 1e-4

    def aa_query(self):
        read_expr = f':A\sX=({fpr})\s+Y=({fpr})\s\\r\\n'
        return Command(send_expr=f'AA X? Y? Z?', read_expr=read_expr).execute(self)

    def aa_set(self, val: int = 85, y: bool = False):
        return Command(send_expr=f'AA Y={val}' if y else f'AA X={val}', read_expr=':A').execute(self)

    def aa(self, y: bool = False):
        return Command(send_expr=f'AA Y' if y else f'AA X', read_expr=':A').execute(self)

    def az(self, y: bool = False):
        return Command(send_expr=f'AZ Y' if y else f'AZ X', read_expr=':A').execute(self)


class Errors(object):
    PI_NO_ERROR = 0
    PI_DEVICE_NOT_FOUND = 1
    PI_OBJECT_NOT_FOUND = 2
    PI_CANNOT_CREATE_OBJECT = 3
    PI_INVALID_DEVICE_HANDLE = 4
    PI_READ_TIMEOUT = 5
    PI_READ_THREAD_ABANDONED = 6
    PI_READ_FAILED = 7
    PI_INVALID_PARAMETER = 8
    PI_WRITE_FAILED = 9


class Command(object):

    def __init__(self, send_expr, read_expr=':A', group=None, timeout=0.0):
        self.send_expr = send_expr
        self.read_expr = read_expr
        self.group = group
        self.timeout = timeout
        self.last_sent = ''

    def execute(self, ser, **kwargs):
        self.write(ser, **kwargs)
        return self.verify(ser)

    def write(self, ser, **kwargs):
        formatted = self.send_expr.format(**kwargs)  # Format the message to send
        ser.write(formatted)
        self.last_sent = formatted

    def verify(self, ser):
        msg, is_match = ser.read_until(expr=self.read_expr, group_num=self.group, timeout=self.timeout)
        if not is_match:
            logger.warning(f'Command {self.last_sent} not validated. The following message was retrieved: {msg}')
        return msg
