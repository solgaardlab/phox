import nidaqmx
import nidaqmx.system

import numpy as np
from typing import List, Dict


class MeshAOControl:
    def __init__(self):
        self.system = nidaqmx.system.System.local()
        self.ao_channels = [channel for device in self.system.devices
                            for channel in device.ao_physical_chans]

    def write_chans(self, channel_to_voltages: Dict[int, np.ndarray]):
        with nidaqmx.Task() as task:
            num_samples = []
            for channel_idx, voltages in channel_to_voltages.items():
                task.ao_channels.add_ao_voltage_chan(self.ao_channels[channel_idx].name)
                num_samples.append(task.write(voltages, auto_start=True))
        return num_samples

    def write_chan(self, channel_idx: int, voltages: np.ndarray):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(self.ao_channels[channel_idx].name)
            num_samples = task.write(voltages, auto_start=True)
        return num_samples
