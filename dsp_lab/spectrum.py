import numpy as np


def dft_matrix(n):
    kn = np.arange(n) * np.arange(n).reshape((-1, 1))
    return np.exp(-1j * 2 * np.pi * kn / n)

def dft(x):
    n = len(x)
    return dft_matrix(n) @ x

def idft(X):
    n = len(X)
    # матрица симметрична, требуется только убрать минус в степени комплексной экспоненты
    M_inv = dft_matrix(n).conj() / n
    return M_inv @ X

def freq_axis(n, fs):
    """
    Частотные значения бинов ДПФ порядка n при частоте дискретизации сигнала fs
    :param n: порядок ДПФ
    :param fs: частота дискретизации сигнала
    :return: значения бинов в диапазоне [0; fs)
    """
    return np.arange(n) * fs / n

def alias_frequency(f, fs):
    f_alias = f % fs
    if f_alias > fs / 2:
        f_alias = -f_alias % fs
    return f_alias


def circular_convolution(x, h):
    n = max(len(x), len(h))
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    if len(h) < n:
        h = np.pad(h, (0, n - len(h)))

    return idft(dft(x) * dft(h))


def linear_convolution_via_dft(x, h):
    L = len(x) + len(h) - 1
    x = np.pad(x, (0, L - len(x)))
    h = np.pad(h, (0, L - len(h)))

    return circular_convolution(x, h)


def normalized_correlation(x, p):
    N = len(x)
    M = len(p)

    is_complex = True
    if not np.iscomplexobj(x) and not np.iscomplexobj(p):
        is_complex = False

    corr = np.zeros(N - M + 1, dtype=float)
    for k in range(N - M + 1):
        x_slice = x[k: k + M]

        numerator = np.sum(x_slice * np.conj(p))
        if is_complex:
            numerator = np.abs(numerator)

        norm_x = np.sum(np.abs(x_slice) ** 2)
        norm_p = np.sum(np.abs(p) ** 2)
        norm = np.maximum(norm_x * norm_p, 1e-12)
        denom = np.sqrt(norm)

        corr[k] = numerator / denom
    return corr


if __name__ == '__main__':
    dft_matrix(4)