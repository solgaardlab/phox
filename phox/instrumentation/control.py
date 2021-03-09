import nidaqmx
import nidaqmx.system

import numpy as np


class MeshAOControl:
    def __init__(self, vmax: float = 6):
        self.system = nidaqmx.system.System.local()
        self.ao_channels = [channel for device in self.system.devices
                            for channel in device.ao_physical_chans]
        self.vmax = vmax

    def write_chan(self, chan: int, voltages: np.ndarray) -> int:
        """Write voltages to channel

        Args:
            chan: Channel to write
            voltages: Voltages sent to channel

        Returns:
            Number of written samples

        """
        if np.sum(voltages > self.vmax) > 0:
            raise ValueError(f'All voltages written to channel must be <= {self.vmax}.')
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(self.ao_channels[chan].name)
            num_samples = task.write(voltages, auto_start=True)
        return num_samples
