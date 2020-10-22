
"""
HEWLETT-PACKARD, 8164A (Tunable Laser Source)
We use GPIB-USB Interface to communicate
"""


import pyvisa
import numpy as np
import time


rm = pyvisa.ResourceManager()
List = rm.list_resources()
print(List)
i=int(input("Port Number:"))
address = List[i]
my_instrument = rm.open_resource(address)

my_instrument.read_termination = '\n'
my_instrument.write_termination ='\r\n'
my_instrument.query('*IDN?')
print(my_instrument.query('*IDN?'))


def power_set(self, power):
    self.write('sour0:pow ' + str(power) + 'mW')

def wavelength_set(self, wavelength):
    self.write('sour0:wav ' +str(wavelength)+ 'NM')

def wavelength_sweep(self, start_wavelength, stop_wavelength, step, speed):
    self.write('wav:swe:star ' + str(start_wavelength) + 'nm')
    self.write('wav:swe:stop ' + str(stop_wavelength) + 'nm')
    self.write('wav:swe:step ' + str(step) + 'nm')
    self.write('wav:swe:spe ' + str(speed) + 'nm/s')
    self.write('wav:swe 1')
    time.sleep(speed)
    my_instrument.write('wav:swe 0')

power_set(my_instrument, 3)
wavelength_sweep(my_instrument, 1530, 1560, 10, 10)
wavelength_set(my_instrument, 1550)
