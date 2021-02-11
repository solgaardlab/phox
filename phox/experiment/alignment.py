# from ..instrumentation import APTPiezo
# from ..instrumentation.serial import SerialMixin
#
#
# class FiberGratingAlignment(SerialMixin):
#     def __init__(self, port: str = '/dev/ttyUSB4', channel_idx: int = 1):
#         self.channel_idx = channel_idx
#         SerialMixin.__init__(self,
#                              port=port,
#                              id_command='*IDN?',
#                              id_response='HP8163A',
#                              terminator='\r'
#                              )
#
#     @property
#     def power(self):
#         return