# # from .serial import SerialMixin
# from pyftdi.ftdi import Ftdi
# import time
#
#
# class APTPiezo:
#     def __init__(self):
#         self.interface = Ftdi()
#         self.interface.open(vendor=0x0403, product=0xfaf0)
#
#         # As protocol describes set the baudrate to 115200,
#         # data definition to 8 data bytes, 1 stop bit, no parity
#         self.interface.set_baudrate(115200)
#         self.interface.set_line_property(8, 1, "N")
#
#         # Pre-purge and post-purge rest, purge RX/TX buffers of the FTDI chip
#         time.sleep(0.05)
#         self.interface.purge_buffers()
#         time.sleep(0.05)
#
#         # Set flow control to RTS/CTS and flag RTS as true
#         self.interface.set_flowctrl("hw")
#         self.interface.set_rts(True)
#
#     def write_data(self, data):
#         self.interface.write_data(data)
#
#     def read_data(self, amount_bytes, retry_attempts):
#         return self.interface.read_data_bytes(amount_bytes, retry_attempts)





