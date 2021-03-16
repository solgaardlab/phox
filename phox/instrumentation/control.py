import nidaqmx
import nidaqmx.system

import numpy as np
from typing import Callable, Optional, Tuple
from nidaqmx.constants import AcquisitionType
import time

class MeshAOControl:
    def __init__(self, vmax: float = 6):
        self.system = nidaqmx.system.System.local()
        self.ao_channels = [channel for device in self.system.devices
                            for channel in device.ao_physical_chans]
        self.vmax = vmax

    def reset(self):
        for device in self.system.devices:
            device.reset_device()

    def write_chan(self, chan: int, voltages: np.ndarray, n_callback: Optional[Tuple[Callable, int]] = None,
                   sweep_time: float = 10) -> int:
        """Write voltages to channel

        Args:
            chan: Channel to write
            voltages: Voltages sent to channel
            n_callback: A tuple of num samples and callback function
            sweep_time: Sweep time in seconds

        Returns:
            Number of written samples

        """
        if np.sum(voltages > self.vmax) > 0:
            raise ValueError(f'All voltages written to channel must be <= {self.vmax}.')

        num_voltages = 1 if not isinstance(voltages, np.ndarray) else voltages.size
        task = nidaqmx.Task()
        task.ao_channels.add_ao_voltage_chan(self.ao_channels[chan].name)
        if n_callback is not None:
            task.timing.cfg_samp_clk_timing(rate=num_voltages / sweep_time, sample_mode=AcquisitionType.CONTINUOUS)
            task.register_every_n_samples_transferred_from_buffer_event(*n_callback)
        num_samples = task.write(voltages)
        task.start()
        time.sleep(num_samples / task.timing.samp_clk_rate)
        task.close()
        return num_samples

