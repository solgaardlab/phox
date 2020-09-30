__author__ = 'Joe Landry'

import serial
import logging
import time
import re
import threading

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

    def __init__(self, lib=None):
        """
        Sets up basic functionality for base communications
        """
        self.lib = lib
        self.lock = threading.Lock()

        defaults = {SerAttr.port: '/dev/ttyUSB0',  # Port (usually 'COMX' on Windows)
                    SerAttr.id_command: "",  # Command used to verify device communication
                    SerAttr.id_response: "",  # Response used to verify device communication
                    SerAttr.default_timeout: 0.1,
                    SerAttr.terminator: '\r',  # Character to send at the end of each line
                    SerAttr.baudrate: 115200,
                    SerAttr.bytesize: 8,
                    SerAttr.parity: 'N',
                    SerAttr.xonxoff: False,
                    SerAttr.rtscts: False,
                    SerAttr.stopbits: 1}

        # self.serial_cfg = lib['Serial'].validateInPlace(defaults)  # Configuration object
        self.serial_cfg = defaults
        self._ser = serial.Serial()  # Serial object provided by pyserial

        self._is_verified = False
        self._maxBuffer = 4096  # Default device output buffer size
        self._isOnline = False  # True when device is verified by instrument returning ID response.
        self._ser.write_timeout = .5  # [s] Write timeout. Necessary for some instruments (depending on handshaking?)

    def open(self):
        """
        1. Applies all settings from config file
        2. Attempts to open the port with the given settings.
        3. Asks for the instrument's ID to make sure it can send/receive commands
        :return: None
        """
        if self._ser.is_open:
            self._ser.close()
        self._ser.port = self.serial_cfg[SerAttr.port]
        self._ser.baudrate = self.serial_cfg[SerAttr.baudrate]
        self._ser.bytesize = self.serial_cfg[SerAttr.bytesize]
        self._ser.parity = self.serial_cfg[SerAttr.parity]
        self._ser.stopbits = self.serial_cfg[SerAttr.stopbits]
        self._ser.xonxoff = self.serial_cfg[SerAttr.xonxoff]
        self._ser.rtscts = self.serial_cfg[SerAttr.rtscts]

        try:
            self._ser.open()
            self.verify()
        except serial.serialutil.SerialException:
            logger.warning('%s could not be opened for %s. Might be off or disconnected.',
                           self._ser.port, self.__class__.__name__)

    def verify(self):
        message, matched = self.send(self._id_command).read_until(self._id_response)
        if matched:
            logger.info('Verified device operation')
            self._is_verified = True
        else:
            self._is_verified = False
            logger.warning('Failed verification. Device might be off.')

    def close(self):
        self._ser.close()

    def send(self, cmd):
        """
        Attempts to send command to device. No command is sent if the port isn't open.
        :param cmd: string; command
        :return: None
        """
        self.lock.acquire()
        self._openCheck()
        self.flush()  # Flush before writing to clear any data in input buffer
        if self._ser.is_open:
            try:
                self._ser.write(f'{cmd}{self._term}'.encode())
                logger.info(f'Sent command: {cmd}')
            except serial.serialutil.SerialTimeoutException:
                logger.warning('Could not send command. Timeout exception.')
        else:
            logger.warning(f'Command "{cmd}" not sent. Port closed.')
        self.lock.release()
        return self

    def read_until(self, expr, group_num=None, timeout=0):
        """
        Reads output until a specific phrase is found. If no timeout is provided, the config timeout is used
        :param expr: string after which to stop reading bytes
        :param group_num: group number
        :param timeout: timeout [s]
        :return: tuple; (message, success); message is the message that was read corresponding to the groupNumber.
        success is whether or not there was a match to the expression.
        """
        self.lock.acquire()
        if not self._ser.is_open:
            self.lock.release()
            return None, False
        timeout = timeout if timeout != 0 else self._timeout  # Use default timeout if one is not provided
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

    def read(self, num_bytes=0, timeout=0):
        """
        Reads specified number of bytes from the stream and then returns. If numBytes is 0, read will timeout after
        the specified timeout. If timeout is 0, the default timeout is used.
        """
        self.lock.acquire()
        if not self._ser.is_open:
            self.lock.release()
            return None
        start_time = time.time()
        message = ''
        total_bytes_read = 0
        timeout = timeout if timeout != 0 else self._timeout  # Set the timeout for this read
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

    def reload(self):
        """
        Reloads configuration file
        :return: None
        """
        self.lib.load()  # Load edited file
        self.open()  # Try to reopen the port with the new settings

    def is_online(self):
        return self._ser.is_open

    def is_verified(self):
        return self._is_verified

    @property
    def _timeout(self):
        return self.serial_cfg[SerAttr.default_timeout]

    @property
    def _term(self):
        return self.serial_cfg[SerAttr.terminator]

    @property
    def _id_command(self):
        return self.serial_cfg[SerAttr.id_command]

    @property
    def _id_response(self):
        return self.serial_cfg[SerAttr.id_response]

    def _openCheck(self):
        if not self._ser.is_open:
            logger.warning('{0} is offline, cannot read.'.format(self.__class__.__name__))
            return False
        return True
