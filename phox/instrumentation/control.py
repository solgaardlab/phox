import nidaqmx
import nidaqmx.system

import numpy as np
from typing import Callable, Optional, Tuple
from nidaqmx.constants import AcquisitionType
import time
import panel as pn


class NIDAQControl:
    def __init__(self, vmin: float = 0, vmax: float = 5):
        self.system = nidaqmx.system.System.local()
        self.ao_channels = [channel for device in self.system.devices
                            for channel in device.ao_physical_chans]
        self.vmax = vmax
        self.vmin = vmin

    def reset(self):
        for device in self.system.devices:
            device.reset_device()

    def ttl_toggle(self, chan: int):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(self.ao_channels[chan].name)
            task.write(5)
            task.write(0)

    def continuous_slider(self, chan: int, name: str, vlim: Tuple[float, float] = (0, 5), default_v: float = 0):
        def change_voltage(*events):
            for event in events:
                if event.name == 'value':
                    with nidaqmx.Task() as task:
                        task.ao_channels.add_ao_voltage_chan(self.ao_channels[chan].name)
                        task.write(event.new)
        voltage = pn.widgets.FloatSlider(start=vlim[0], end=vlim[1], step=0.01,
                                         value=default_v, name=name, format='1[.]000')
        voltage.param.watch(change_voltage, 'value')
        return voltage

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
        if np.sum(voltages < self.vmin) > 0:
            raise ValueError(f'All voltages written to channel must be >= {self.vmin}.')

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

