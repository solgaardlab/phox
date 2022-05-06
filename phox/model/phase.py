import numpy as np
from typing import Tuple, Optional
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt


class PhaseCalibration:
    def __init__(self, vs: np.ndarray, powers: np.ndarray, spot: Tuple[int, int],
                 coefficients: Optional[np.ndarray] = None, a: Optional[float] = None, b: Optional[float] = None,
                 p0: Tuple[float, ...] = (1, 0, 0, 0.3, 0, 0)):
        """Phase calibration object consisting of voltages and camera spot power measurements,
        We assume that light enters the lower (smaller idx) input.

        Args:
            vs: voltages
            powers: camera spot powers (tuple of 18 spot powers for a 6x3 array... currently hardcoded!)
            spot: Tuple of layer, index of the waveguide mode for the bottom port of the calibration
            coefficients: p and q coefficients for the polynomial fit for phase v voltage and voltage v phase
            a: amplitude for the calibration
            b: offset for the calibration
            p0: initial guess for the calibration of the phase shifter
        """
        self.vs = vs
        self.vmin = np.min(self.vs)
        self.vmax = np.max(self.vs)
        self.powers = np.fliplr(powers.reshape((6, 3, -1)))
        self.spot = spot
        self.idx = spot[1]
        if coefficients is None:
            fit = self.fit(p0)
            self.coefficients, self.a, self.b = fit[0], fit[1], fit[2]  # tuple assignment being weird?
        else:
            self.coefficients, self.a, self.b = coefficients, a, b
        vs, sr = self.vs, self.lower_split_ratio
        self.top_peaks = find_peaks_cwt(sr, widths=[800])[1]
        self.bottom_peaks = find_peaks_cwt(-sr, widths=[800])

    def fit(self, p0: Tuple[float, ...], tol: float = 0.01,
            overwrite_shift: bool = True) -> Tuple[np.ndarray, float, float]:
        # automatically find the first lower peak
        lower_v = self.vs[find_peaks_cwt(-self.lower_split_ratio, widths=[800])[0]]

        p0_ = (*p0[:-1], -lower_v) if overwrite_shift else p0
        print(f'Initial params: {p0_}')
        p, pcov = curve_fit(cal_v_power, self.vs, self.lower_split_ratio, p0=p0_)
        error = np.sqrt(np.sum(np.diag(pcov)))

        if error > 3 * tol:
            print(f'Trying some more initial conditions... error {error} is >3x more than the tolerable error {tol}')

        delta = 0
        while error > 3 * tol and p0_[-1] < 0 and delta < 2 * np.pi:
            delta += 0.05
            p, pcov = curve_fit(cal_v_power, self.vs, self.lower_split_ratio, p0=(*p0_[:-1], p0_[-1] + delta))
            error = np.sqrt(np.sum(np.diag(pcov)))

        delta = 0
        # exhaust some more initial conditions before going to the next step
        while error > 3 * tol and p0_[-1] > -5 and delta < -2 * np.pi:
            delta -= 0.05
            p, pcov = curve_fit(cal_v_power, self.vs, self.lower_split_ratio, p0=(*p0_[:-1], p0_[-1] + delta))
            error = np.sqrt(np.sum(np.diag(pcov)))

        print(f'Final p: {p}')

        if error > 3 * tol:
            raise RuntimeError(f'FATAL: The error {error} is too big, need to recalibrate...')
        if error > tol:
            print(f'WARNING: final error {error} is more than the tolerable error {tol}')
        print(f'Fit v2p function with error {error}!')

        q, qcov = curve_fit(cal_phase_v, np.polyval(p[2:], self.vs), self.vs ** 2)
        qerror = np.sqrt(np.sum(np.diag(qcov)))
        print(f'Fit p2v function with error {qerror}!')

        print(f'Final q: {q}')

        if error > 3 * tol:
            raise RuntimeError(f'FATAL: The error {error} is too big, need to recalibrate...')

        # ensure all curves are in the approx the same voltage range
        vmin = np.sqrt(np.abs(np.polyval(q, 0)))
        vmax = np.sqrt(np.abs(np.polyval(q, np.pi)))
        print(f'The 0π, 2π voltages are now {vmin, vmax}')
        if vmax > self.vmax:
            p[-1] += np.pi
            q, _ = curve_fit(cal_phase_v, np.polyval(p[2:], self.vs), self.vs ** 2)
            vmin = np.sqrt(np.abs(np.polyval(q, 0)))
            vmax = np.sqrt(np.abs(np.polyval(q, np.pi)))
            print(f'The 0π, 2π voltages are now {vmin, vmax}')
        return np.vstack([p[2:], q]), p[0], p[1]

    @property
    def lower_split_ratio(self) -> np.ndarray:
        return self.powers[self.idx, 2] / (self.powers[self.idx + 1, 2] + self.powers[self.idx, 2])

    @property
    def upper_split_ratio(self) -> np.ndarray:
        return 1 - self.lower_split_ratio

    @property
    def split_ratio_fit(self) -> np.ndarray:
        return cal_v_power(self.vs, self.a, self.b, *self.coefficients[0])

    @property
    def upper_out(self) -> np.ndarray:
        return self.powers[self.idx + 1, 2] / self.powers[self.idx, 0]

    @property
    def lower_out(self) -> np.ndarray:
        return self.powers[self.idx, 2] / self.powers[self.idx, 0]

    @property
    def upper_arm(self) -> np.ndarray:
        return self.powers[self.idx, 1] / (self.powers[self.idx, 1] + self.powers[self.idx + 1, 1])

    @property
    def lower_arm(self) -> np.ndarray:
        return 1 - self.upper_arm

    @property
    def total_arm(self) -> np.ndarray:
        return (self.powers[self.idx, 1] + self.powers[self.idx + 1, 1]) / (self.powers[self.idx, 0])

    @property
    def total_out(self) -> np.ndarray:
        return (self.powers[self.idx, 2] + self.powers[self.idx + 1, 2]) / (self.powers[self.idx, 0])

    def raw(self, bottom=False, idx=0) -> np.ndarray:
        return self.powers[self.idx + bottom, idx]

    @property
    def dict(self) -> dict:
        return {
            'vs': self.vs,
            'powers': np.fliplr(self.powers),
            'coefficients': self.coefficients,
            'spot': self.spot,
            'a': self.a,
            'b': self.b
        }

    def v2p(self, voltage: float) -> np.ndarray:
        """Voltage to phase conversion for a give phase shifter

        Args:
            voltage: voltage to convert

        Returns:
            Phase converted from voltage

        """
        p, _ = self.coefficients
        return np.polyval(p, voltage) * 2

    def p2v(self, phase: float) -> np.ndarray:
        """Phase to voltage conversion for a give phase shifter

        Args:
            phase: phase to convert

        Returns:
            Voltage converted from phase

        """
        _, q = self.coefficients
        return np.sqrt(np.abs(np.polyval(q, phase / 2)))  # abs suppresses in case negative value

    def p2v_lookup(self, phase):
        sr = self.lower_split_ratio
        idx_pi = self.top_peaks[0]
        idx_0, idx_2pi = self.bottom_peaks[0], self.bottom_peaks[1]
        if isinstance(phase, np.ndarray):
            phase_pi = phase[phase >= np.pi]
            phase_0 = phase[phase < np.pi]
            i_0 = np.argmin(
                np.abs(np.sin(phase_0[:, np.newaxis] / 2) ** 2 - sr[idx_0:idx_pi][np.newaxis, :]), axis=1)
            i_pi = np.argmin(
                np.abs(np.sin(phase_pi[:, np.newaxis] / 2) ** 2 - sr[idx_pi:idx_2pi][np.newaxis, :]), axis=1)
            v_0 = self.vs[idx_0 + i_0]
            v_pi = self.vs[idx_pi + i_pi]
            v = np.zeros_like(phase)
            v[phase >= np.pi] = v_pi
            v[phase < np.pi] = v_0
        else:
            if 0 <= phase < np.pi:
                v = self.vs[idx_0 + np.argmin(np.abs(np.sin(phase / 2) ** 2 - sr[idx_0:idx_pi]))]
            else:
                v = self.vs[idx_pi + np.argmin(np.abs(np.sin(phase / 2) ** 2 - sr[idx_pi:idx_2pi]))]
        return v


def cal_v_power(v, a, b, p0, p1, p2, p3):
    # take the absolute value of a to ensure a is positive!
    # need to also take absolute value of p3
    # ensure that the shifted phase always goes from +0 to +np.pi to ensure it is within desired range
    return np.abs(a) * np.sin(p0 * v ** 3 + p1 * v ** 2 + p2 * v - np.abs(p3)) ** 2 + b


def cal_phase_v(x, q0, q1, q2, q3):
    return q0 * x ** 3 + q1 * x ** 2 + q2 * x + q3
