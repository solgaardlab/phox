__author__ = 'Joe Landry'

import serial
import logging
import time
import re
import threading

from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)


class SerAttr(object):
    """
    Enum for base attributes
    """

    port = 'port'
    id_command = 'id_command'
    id_response = 'id_response'
    default_timeout = 'default_timeout'
    terminator = 'terminator'
    baudrate = 'baudrate'
    bytesize = 'bytesize'
    parity = 'parity'
    xonxoff = 'xonxoff'
    rtscts = 'rtscts'
    stopbits = 'stopbits'


class SerialMixin(object):
    """
    Serial: Base class for all base communications based devices. Handles opening
    of ports, as well as reading and writing.
    """

    def __init__(self, port: str = '/dev/ttyUSB0', id_command: str = '', id_response: str = '',
                 default_timeout: float = 0.1, terminator: str = '\r', baudrate: int = 115200,
                 bytesize: int = 8, parity: str = 'N', xonxoff: bool = False, rtscts: bool = False, stopbits: int = 1):
        """
        Sets up basic functionality for base communications
        """
        self.lock = threading.Lock()

        self.port = port  # Port (usually 'COMX' on Windows)
        self.id_command = id_command  # Command used to verify device communication
        self.id_response = id_response  # Response used to verify device communication
        self.default_timeout = default_timeout
        self.terminator = terminator  # Character to send at the end of each line
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.xonxoff = xonxoff
        self.rtscts = rtscts
        self.stopbits = stopbits
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
            xonxoff=xonxoff,
            rtscts=rtscts
        )  # Serial object provided by pyserial

        self._is_verified = False
        self._maxBuffer = 4096  # Default device output buffer size
        self._isOnline = False  # True when device is verified by instrument returning ID response.
        self._ser.write_timeout = .5  # [s] Write timeout. Necessary for some instruments (depending on handshaking?)

    def open(self):
        """
        1. Attempts to open the port with the given settings.
        2. Asks for the instrument's ID to make sure it can send/receive commands
        :return: None
        """
        if self._ser.is_open:
            self._ser.close()
        try:
            self._ser.open()
            self.verify()
        except serial.serialutil.SerialException:
            logger.warning('%s could not be opened for %s. Might be off or disconnected.',
                           self._ser.port, self.__class__.__name__)

    def verify(self):
        message, matched = self.write(self.id_command).read_until(self.id_response)
        if matched:
            logger.info('Verified device operation')
            self._is_verified = True
        else:
            self._is_verified = False
            logger.error('Failed verification. Device might be off.')

    def close(self):
        self._ser.close()

    def write(self, cmd: str):
        """
        Attempts to send command to device. No command is sent if the port isn't open.
        :param cmd: string; command
        :return: None
        """
        self.lock.acquire()
        self._open_check()
        self.flush()  # Flush before writing to clear any data in input buffer
        if self._ser.is_open:
            try:
                self._ser.write(f'{cmd}{self.terminator}'.encode())
                logger.info(f'Sent command: {cmd}')
            except serial.serialutil.SerialTimeoutException:
                logger.warning('Could not send command. Timeout exception.')
        else:
            logger.warning(f'Command "{cmd}" not sent. Port closed.')
        self.lock.release()
        return self

    def read_until(self, expr: str, group_num: int = None, timeout: int = 0) -> Tuple[
        Optional[Union[str, tuple]], bool]:
        # """
        # Reads output until a specific phrase is found. If no timeout is provided, the config timeout is used
        # :param expr: string after which to stop reading bytes
        # :param group_num: group number
        # :param timeout: timeout [s]
        # :return: tuple; (message, success); message is the message that was read corresponding to the groupNumber.
        # success is whether or not there was a match to the expression.
        # """
        """Reads output until a specific phrase is found. If no timeout is provided, the config timeout is used

        Args:
            expr: string after which to stop reading bytes
            group_num: group number
            timeout: timeout [s]

        Returns:
            (message, success); message is the message that was read corresponding to the groupNumber.
            success is whether or not there was a match to the expression.

        """
        self.lock.acquire()
        if not self._ser.is_open:
            self.lock.release()
            return None, False
        timeout = timeout if timeout != 0 else self.default_timeout  # Use default timeout if one is not provided
        message = ''
        if self._ser.is_open:
            start_time = time.time()
            while time.time() - start_time < timeout:
                message = f'{message}{self._ser.read(self._ser.in_waiting).decode("utf-8")}'
                match = re.search(expr, message)
                if match is not None:
                    logger.debug(f"Matched expression: {expr}")
                    if group_num is None:
                        self.lock.release()
                        return message, True
                    elif isinstance(group_num, (list, tuple)):
                        self.lock.release()
                        return tuple(match.group(i) for i in group_num), True
                    else:
                        self.lock.release()
                        return match.group(group_num), True
            logger.warning(f'No regex match for: {expr}')
            self.lock.release()
            return message, False

    def setup(self):
        pass

    def read(self, num_bytes: int = 0, timeout: int = 0):
        """Reads specified number of bytes from the stream and then returns. If numBytes is 0, read will timeout after
        the specified timeout. If timeout is 0, the default timeout is used.

        Args:
            num_bytes:
            timeout:

        Returns:

        """
        self.lock.acquire()
        if not self._ser.is_open:
            self.lock.release()
            return None
        start_time = time.time()
        message = ''
        total_bytes_read = 0
        timeout = timeout if timeout != 0 else self.default_timeout  # Set the timeout for this read
        while time.time() - start_time < timeout:
            bytes_waiting = self._ser.in_waiting
            if (bytes_waiting > (num_bytes - total_bytes_read)) & (num_bytes != 0):
                message += self._ser.read(num_bytes - total_bytes_read)
                self.lock.release()
                return message
            else:
                message += self._ser.read(bytes_waiting)
                total_bytes_read += bytes_waiting
        self.lock.release()
        return message

    def flush(self):
        """
        Simply flushes the input buffer (the output of the device) so it isn't read during subsequent calls
        :return: None
        """
        if self._ser.is_open:
            self._ser.flushInput()
            logger.debug('Flushed input buffer.')

    def is_online(self):
        return self._ser.is_open

    def is_verified(self):
        return self._is_verified

    def _open_check(self):
        if not self._ser.is_open:
            logger.warning('{0} is offline, cannot read.'.format(self.__class__.__name__))
            return False
        return True

    def connect(self):
        if not self.is_online():
            self.open()
        else:
            self.verify()
        if self.is_verified():
            self.setup()
        return self
