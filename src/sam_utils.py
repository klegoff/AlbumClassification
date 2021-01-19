# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_sound(x, fs, **kwargs):
    """
    Display a sound (waveform) as a function of the time in seconds.

    Parameters
    ----------
    x : ndarray
        Sound to be displayed
    fs : int or float
        Sampling frequency
    kwargs
        Any optional argument passed to the ``matplotlib.pyplot.plot``
        function.
    """
    t = np.arange(x.shape[0]) / fs
    plt.plot(t, x, **kwargs)
    plt.xlabel('time (s)')


def db(x):
    """
    Conversion to decibels

    Parameters
    ----------
    x : ndarray
        Input array to be converted

    Returns
    -------
    ndarray
        The result is an array with same shape as ``x`` and values obtained by
        applying 20*log10(abs(.)) to each coefficient in ``x``
    """
    return 20 * np.log10(np.abs(x))


def plot_spectrum(x, fs=1, n_fft=None, fft_shift=False, **kwargs):
    if n_fft is None:
        n_fft = x.shape[0]
    X = np.fft.fft(x, n=n_fft)
    if fft_shift:
        X = np.fft.fftshift(X)
        f_range = np.fft.fftshift(np.fft.fftfreq(n_fft) * fs)
    else:
        f_range = np.arange(n_fft) / n_fft * fs
    plt.plot(f_range, db(X), **kwargs)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectrum (dB)')


def show_spectrum_2d(img, db_scale=False):
    """
    Display the 2D-spectrum of an image

    Parameters
    ----------
    img : ndarray (2d)
        Image
    db_scale : bool
        If true, values are displayed in decibels. If False, display the
        modulus of the complex values.

    """
    N, M = img.shape
    S = np.fft.fft2(img)
    if db_scale:
        plt.imshow(db(S), extent=(-0.5/M, 1-0.5/M, 1-0.5/N, -0.5/N))
    else:
        plt.imshow(np.abs(S), extent=(-0.5/M, 1-0.5/M, 1-0.5/N, -0.5/N))
    plt.colorbar()


def add_noise(x, snr=20):
    n = np.random.randn(*x.shape)
    n *= 10**(-snr/20) * np.linalg.norm(x) / np.linalg.norm(n)
    return x + n


def snr(x_ref, x_est):
    return 20 * np.log10(np.linalg.norm(x_ref) / np.linalg.norm(x_ref - x_est))
