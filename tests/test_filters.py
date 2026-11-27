import numpy as np

from dsp_lab.filters import fir_lowpass
import scipy.signal as sg
import pytest


@pytest.mark.parametrize(
    ("numtaps", "cutoff", "fs", "window"),
    [
        (32, 1000, 8000, "rect"),
        (32, 1000, 8000, "hann"),
        (32, 1000, 8000, "hamming"),
        (32, 1000, 8000, "blackman"),
        (65, 1000, 8000, "rect"),
        (65, 1000, 8000, "hann"),
        (65, 1000, 8000, "hamming"),
        (65, 1000, 8000, "blackman"),
        (32, 0.1, 1, "blackman"),
    ]
)
def test_fir_lowpass(numtaps, cutoff, fs, window):
    h = fir_lowpass(numtaps, cutoff, fs, window)
    h_expected = sg.firwin(numtaps, cutoff, window=window, fs=fs)

    assert np.allclose(h, h_expected)