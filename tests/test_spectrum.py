import pytest
import numpy as np
from dsp_lab.spectrum import dft_matrix, dft, idft, freq_axis, alias_frequency, linear_convolution_via_dft, \
    circular_convolution

rng = np.random.default_rng(seed=666)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 1024])
def test_dft_matrix_shape(n: int):
    M = dft_matrix(n)
    assert M.shape == (n, n)


def test_dft_matrix_values():
    n = 16
    M = dft_matrix(n)
    for i in range(n):
        for j in range(n):
            assert np.isclose(M[i, j], np.exp(-1j * 2 * np.pi * i * j / n))


@pytest.fixture
def x(n):
    rng = np.random.default_rng(seed=666)
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)

@pytest.fixture
def X(n):
    rng = np.random.default_rng(seed=666)
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 1024])
def test_dft(x):
    X_actual = dft(x)
    X_expected = np.fft.fft(x)

    assert np.allclose(X_actual, X_expected)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 1024])
def test_idft(X):
    x_actual = idft(X)
    x_expected = np.fft.ifft(X)

    assert np.allclose(x_actual, x_expected)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 1024])
def test_idft_normalization(x):
    X = dft(x)
    x_actual = idft(X)

    assert np.allclose(x_actual, x)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 1024])
def test_parseval(x):
    X = dft(x)
    n = len(x)

    assert np.isclose(
        np.sum(np.abs(x)**2),
        np.sum(np.abs(X)**2) / n
    )


@pytest.fixture
def x_real(n):
    rng = np.random.default_rng(seed=666)
    return rng.standard_normal(n)

@pytest.mark.parametrize("n", [3, 4, 8, 511, 1024])
def test_conj_symmetry(x_real):
    X = dft(x_real)
    print(X)

    n = len(x_real)
    # пропускаем X_0 (постоянную составляющую)
    # сравниваем X_1 с X_{N-1}
    for k in range(1, n):
        assert np.isclose(X[k], np.conj(X[n - k]))

@pytest.mark.parametrize("n", [1, 2, 3, 31, 64])
def test_freq_axis(n):
    fs = 1000

    f_actual = freq_axis(n, fs)
    f_expected = np.fft.fftfreq(n, 1 / fs)

    assert np.allclose(f_actual[:n // 2], f_expected[:n // 2])
    assert np.allclose(f_actual[n // 2 + n % 2:], f_expected[n // 2 + n % 2:] + fs)

@pytest.mark.parametrize("n", [1, 2, 3, 31, 64])
def test_freq_axis_len(n):
    f = freq_axis(n, 44100)
    assert len(f) == n


@pytest.mark.parametrize(
    ("f", "alias"),
    [
        (20000, 20000),
        (64100, 20000),
        (30000, 14100),
        (74100, 14100),
        (118200, 14100),
        (45100, 1000),
        (89200, 1000),
        (-3000, 3000),
        (-30000, 14100),
        (-30000, 14100),
        (-74100, 14100),
        (22050, 22050),
        (44100, 0),
    ]
)
def test_alias_frequency(f, alias):
    f_alias = alias_frequency(f, 44100)
    assert np.isclose(f_alias, alias)


def complex_signal(n):
    rng = np.random.default_rng(seed=666)
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)


@pytest.mark.parametrize(
    ("n1", "n2"),
    [
        (1, 1),
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4),
        (511, 256),
        (256, 511)
    ]
)
def test_circular_convolution(n1, n2):
    x = complex_signal(n1)
    h = complex_signal(n2)
    N = max(n1, n2)

    y_actual = circular_convolution(x, h)
    y_expected = np.fft.ifft(np.fft.fft(x, n=N) * np.fft.fft(h, n=N))

    assert np.allclose(y_actual, y_expected)


@pytest.mark.parametrize(
    ("n1", "n2"),
    [
        (1, 1),
        (4, 4),
        (4, 3),
        (3, 4),
        (511, 321)
    ]
)
def test_linear_convolution_via_dft(n1, n2):
    x = complex_signal(n1)
    h = complex_signal(n2)

    y_actual = linear_convolution_via_dft(x, h)
    y_expected = np.convolve(x, h, mode="full")

    assert np.allclose(y_actual, y_expected)

