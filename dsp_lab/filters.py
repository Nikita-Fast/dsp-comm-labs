import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt


def fir_lowpass(numtaps: int, cutoff: float, fs: float, window: str = "hamming") -> np.ndarray:
    """
    Расчет КИХ ФНЧ методом окна. Частота среза cutoff — в Гц, частота на которой ослабление составляет -6 дБ.
    """
    if cutoff / fs > 0.5:
        raise ValueError("Частота среза должна быть от 0 до 0.5*fs")

    n = np.arange(numtaps) - (numtaps - 1) / 2
    h = np.sinc(2 * n * cutoff / fs)

    # Грабля: не указать fftbins=False и получить не симметричное окно
    w = sg.get_window(window, numtaps, fftbins=False)
    h_windowed = h * w

    return h_windowed / np.sum(h_windowed)


def freq_response(b: np.ndarray, a: np.ndarray, fs: float, n: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает f (Гц) и H(e^{j2πf/Fs}), посчитанные руками."""
    pass
# в каких пределах функция возвращает результат? для каких частот?

