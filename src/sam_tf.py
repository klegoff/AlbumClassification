# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings(action="ignore",
                        category=RuntimeWarning,
                        message='divide by zero encountered in log10',
                        module='sam_tf')


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

