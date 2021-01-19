# -*- coding: utf-8 -*-
"""

sam_io

Ce fichier permet de lire et écrire des fichiers wav standards, c'est-à-dire
dont les échantillons sont codés en int16, tout en manipulant des signaux
dont les échantillons sont codés en float dans [-1, 1]. Les fonctions de
lecture `read_wav` et écriture `write_wav` font la conversion.

À utiliser sur les sons du fichiers sons_int.zip

.. date:: 2019-12-09
.. moduleauthor:: Valentin Emiya
"""
import numpy as np
from scipy.io.wavfile import read, write


def read_wav(filename):
    """
    Read a wavefile coded with int16 values and convert it to float values
    in [-1, 1]

    Parameters
    ----------
    filename

    Returns
    -------
    fs
    x
    """
    fs, x = read(filename=filename)
    x = x / 2 ** 15
    return fs, x


def write_wav(filename, x, fs):
    """
    Convert a signal coded in float values to int16 values and save it to a
    wav file.
    Parameters
    ----------
    filename
    x
    fs

    Returns
    -------

    """
    x_norm = 2 ** 15
    m = np.max(np.abs(x))
    if m > 1:
        x_norm = x_norm / m
    x = x * x_norm
    x = x.astype(np.int16)
    write(filename=filename, data=x, rate=fs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fs, x = read_wav('../../data/sons_int/35.wav')
    print(fs, x.shape, x[:5])
    write_wav('tmp.wav', x, fs)
    plt.plot(x)
    plt.show()
