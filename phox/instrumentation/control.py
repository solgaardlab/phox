import nidaqmx
import nidaqmx.system

import numpy as np
from typing import Callable, Optional, Tuple
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
import time
import panel as pn


class NIDAQControl:
    def __init__(self, vmin: float = 0, vmax: float = 5, sample_rate: int = 10000, num_samples: int = 1000, channels: Tuple = (28, 24, 16, 20)):
        self.system = nidaqmx.system.System.local()
        self.ao_channels = [channel for device in self.system.devices
                            for channel in device.ao_physical_chans]
        self.ai_channels = [channel for device in self.system.devices
                            for channel in device.ai_physical_chans]
        self.vmax = vmax
        self.vmin = vmin
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.oscope = nidaqmx.Task()
        for chan in channels:
            self.oscope.ai_channels.add_ai_voltage_chan(self.ai_channels[chan].name, max_val=2, min_val=0)
        self.oscope.timing.cfg_samp_clk_timing(
            rate=sample_rate, 
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS, 
            samps_per_chan=num_samples)
        self.meas_buffer = np.zeros((len(channels), num_samples), np.float64)
        reader = AnalogMultiChannelReader(self.oscope.in_stream)
        def oscope_cb(*args):
            reader.read_many_sample(self.meas_buffer, num_samples)
            return 0
        self.oscope.register_every_n_samples_acquired_into_buffer_event(sample_interval=num_samples, callback_method=oscope_cb)
        # self.oscope.start()

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
        # voltage.param.watch(change_voltage, 'value')
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

    def read_chan(self, chan: int, num_voltages: int, rate: float = 100000, average: bool = True) -> int:
        """Read voltages from channel

        Args:
            chan: Channel to write
            num_voltages: Number of voltages to read
            rate: Number of voltages read per second
            average: Whether to average the voltages read out

        Returns:
            Number of written samples

        """
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(self.ai_channels[chan].name)
            task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.CONTINUOUS)
            voltages = task.read(num_voltages)
        return np.mean(voltages) if average else voltages

