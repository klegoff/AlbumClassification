# -*- coding: utf-8 -*-
"""
moduleauthor: Valentin Emiya
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io.wavfile import read, write
import warnings
from pathlib import Path
import pandas as pd



#############################################
#
#				Signal processing
#
#############################################

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



warnings.filterwarnings(action="ignore",
                        category=RuntimeWarning,
                        message='divide by zero encountered in log10',
                        module='sam_utils')


def compute_stft(x, fs, **stft_params):
    seg_len = int(stft_params['seg_dur'] * fs)
    noverlap = int(stft_params['overlap_ratio'] * seg_len)
    if stft_params['nfft'] is None:
        nfft = 2**np.ceil(np.log2(seg_len)+1)
    else:
        nfft = stft_params['nfft']
    window = stft_params['window']
    f, t, X = stft(x, fs=fs, window=window, nperseg=seg_len,
                   noverlap=noverlap, nfft=nfft, detrend=False,
                   return_onesided=True, boundary='zeros', padded=True,
                   axis=-1)
    return f, t, X


def show_spectrogram(f, t, X, dynrange_db=100):
    X = 20*np.log10(np.abs(X))
    X_max = np.max(X)
    plt.imshow(X, origin='lower', extent=(t[0], t[-1], f[0], f[-1]),
               aspect='auto', vmax=X_max, vmin=X_max-dynrange_db)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')




#############################################
#
#				WAV files IO
#
#############################################

"""
Lire et écrire des fichiers wav standards, c'est-à-dire
dont les échantillons sont codés en int16, tout en manipulant des signaux
dont les échantillons sont codés en float dans [-1, 1]. Les fonctions de
lecture `read_wav` et écriture `write_wav` font la conversion.

À utiliser sur les sons du fichiers sons_int.zip
"""


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






#############################################
#
#				MSDI files IO
#
#############################################


_msdi_path = 'msdi'  # Change this to configure your path to MSDI dataset


def get_msdi_dataframe(msdi_path=_msdi_path):
	return pd.read_csv(Path(msdi_path) / 'msdi_mapping.csv')


def load_mfcc(entry, msdi_path=_msdi_path):
x = np.load(Path(msdi_path) / entry['mfcc'])
	return x[entry['msd_track_id']]


def load_img(entry, msdi_path=_msdi_path):
	return plt.imread(Path(msdi_path) / entry['img'])


def load_deep_audio_features(entry, msdi_path=_msdi_path):
	subset_file = 'X_{}_audio_MSD-I.npy'.format(entry['set'])
	x = np.load(Path(msdi_path) / 'deep_features' / subset_file, mmap_mode='r')
	idx = entry['deep_features']
	return x[idx, :]


def get_set(entry):
	return entry['set']


def get_label(entry):
	return entry['genre']


def get_label_list(msdi_path=_msdi_path):
	df = pd.read_csv(Path(msdi_path) / 'labels.csv', header=None)
	return list(df.iloc[:, 0])


if __name__ == '__main__':
	# MSDI IO example
	msdi = get_msdi_dataframe(_msdi_path)
	print('Dataset with {} entries'.format(len(msdi)))
	print('#' * 80)
	print('Labels:', get_label_list())
	print('#' * 80)

	entry_idx = 23456
	one_entry = msdi.loc[entry_idx]
	print('Entry {}:'.format(entry_idx))
	print(one_entry)
	print('#' * 80)
	mfcc = load_mfcc(one_entry, _msdi_path)
	print('MFCC shape:', mfcc.shape)
	img = load_img(one_entry, _msdi_path)
	print('Image shape:', img.shape)
	deep_features = load_deep_audio_features(one_entry, _msdi_path)
	print('Deep features:', deep_features.shape)
	print('Set:', get_set(one_entry))
	print('Genre:', get_label(one_entry))


	# WAV IO example
	fs, x = read_wav('../../data/sons_int/35.wav')
	print(fs, x.shape, x[:5])
	write_wav('tmp.wav', x, fs)
	plt.plot(x)
	plt.show()


