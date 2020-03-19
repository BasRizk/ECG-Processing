# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
"""
Apply your function to the ECG signal provided in the file “DataN.txt”.
The sampling rate of this ECG signal is 256 Hz. You will need to suggest a
method to compute the threshold needed for detection.


TIPS:
    - The sampling rate of this ECG signal is 256 Hz.
    - N should be set approximately the same as the widest possible QRS
complex
    - 
"""

def qrs_detect(raw_signal, win_size=0):
    """

    Parameters
    ----------
    raw_signal : TYPE
        DESCRIPTION.
    win_size : TYPE
        the moving average window size N.

    Returns
    -------
    a vector that contains the timestamps of the R wave and a vector that
    contains the corresponding RR intervals

    """
    raw_signal = np.array(raw_signal)
    
    plot(raw_signal, title="Raw Signal")
    
    noise_filtered_signal = remove_noise(raw_signal)
    
    diff = differentiate(noise_filtered_signal)
    plot(diff, title="Diff Signal")
    
    sqrd = square(diff)
    smoothed = smooth(sqrd)
    thresholded = threshold(smoothed)
    rr_intervals = rr_define(thresholded)
    return rr_intervals

def remove_noise(sig):
    filtered_signal = notch_filter(sig)
    filtered_signal = bandpass_filter(filtered_signal, 0.1, 45, 5)        #bandpass(signal, lowcut, highcut, order)
    return filtered_signal

def notch_filter(sig):
    fs = 256.0
    f0 = 50.0
    Q = 30.0
    b, a = signal.iirnotch(f0, Q, fs)
    filtered_signal = signal.lfilter(b, a, sig)
    return filtered_signal

def bandpass_filter(sig, low_freq, high_freq, order):
    fs = 256.0
    nyquist_freq = fs * 0.5
    low = low_freq / nyquist_freq
    high = high_freq / nyquist_freq
    b, a = signal.butter(order, [low, high], 'band')
    filtered_signal = signal.lfilter(b, a, sig)
    return filtered_signal

def differentiate(sig):
    # diff_signal = signal.copy()
    t_0 = sig[:-4]
    t_1 = sig[1:-3]
    # T_2 = signal[2:-2]
    t_3 = sig[3:-1]
    t_4 = sig[4:]
    
    sampling_interval = (1/256)
    diff_signal = (1/(8*sampling_interval))*\
        (-t_0 - (2*t_1) + (2*t_3) + (t_4))
    return diff_signal

def square(sig):
    # TODO
    pass

def smooth(sig):
    # TODO
    # smooth the squared signal using a moving average window
    
    pass

def threshold(sig):
    # TODO
    pass

def rr_define(rr_intervals):
    # TODO
    pass

def plot(sig, title = "Plot of CT signal"):
    t = np.linspace(-0.02, 0.05, sig.shape[0])
    plt.plot(t, sig)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title(title)
    plt.xlim([-0.02, 0.05])
    plt.show()


if __name__ == '__main__':
    raw_signal = pd.read_csv("DataN.txt", header=None)[0]
    rr_graph = qrs_detect(raw_signal) 
        
    
    


