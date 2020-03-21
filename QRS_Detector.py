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

def qrs_detect(raw_signal, win_size=15):
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
    plot(noise_filtered_signal, title="Noise Removed Signal")

    diff = differentiate(noise_filtered_signal)
    plot(diff, title="Differentiated Signal")
    
    sqrd = square(diff)
    plot(sqrd, title="Squared Signal")

    smoothed = smooth(sqrd, win_size)
    plot(smoothed, title="Smoothed Signal")

    thresholded = threshold(smoothed)
    plot(thresholded, title="Thresholded Signal")

    rr_intervals = rr_define(thresholded)
    plot(rr_intervals, title="RR-Intervals", sampling_rate=1)

    return rr_intervals

def remove_noise(sig):
    filtered_signal = notch_filter(sig, 50.0)
    filtered_signal = bandpass_filter(filtered_signal, 0.1, 45.0, 5)       
    #bandpass(signal, lowcut, highcut, order)
    return filtered_signal


def notch_filter(sig, cut_freq):
    fs = 256.0
    Q = 30.0
    nyquist_freq = fs * 0.5
    cut = cut_freq / nyquist_freq
    numerator, denominator = signal.iirnotch(cut, Q)
    filtered_signal = signal.lfilter(numerator,denominator, sig)
    return filtered_signal
    

def bandpass_filter(sig, low_freq, high_freq, order):
    fs = 256.0
    nyquist_freq = fs * 0.5
    low = low_freq / nyquist_freq
    high = high_freq / nyquist_freq
    numerator, denominator = signal.butter(order, [low, high], 'band')
    filtered_signal = signal.lfilter(numerator, denominator, sig)
    return filtered_signal

def differentiate(sig):
    t_0 = sig[:-4]
    t_1 = sig[1:-3]
    t_3 = sig[3:-1]
    t_4 = sig[4:]
    
    sampling_interval = (1/256)
    diff_signal = (1/(8*sampling_interval))*\
        (-t_0 - (2*t_1) + (2*t_3) + (t_4))
    return diff_signal

# def differentiate2(sig):
#     diff_signal = np.zeros(sig.shape)
#     for i in range(2, len(sig)-2):
#         diff_signal[i] = (1/8) * (-sig[i-2] - 2*sig[i-1] + 2*sig[i+1] + sig[i+2])
#     return diff_signal


def square(sig):
    return np.square(sig)



def smooth(sig, win_size=5):
    # smooth the squared signal using a moving average window
    smoothed_signal = np.zeros(sig.shape)
    for i in range(win_size - 1, len(sig)):
        avg_value = 0
        for v_i in range(win_size):
            avg_value += sig[i-v_i]
        avg_value /= win_size
        smoothed_signal[i] = avg_value
        # print("sig %s ...... smoothed_signal %s" % (sig[i], smoothed_signal[i]))
    return smoothed_signal

def threshold(sig):
    threshold = np.average(sig)*1.5
    # threshold = np.max(sig)*0.7
    thresholded = sig.copy()
    thresholded[thresholded > threshold] = threshold
    return thresholded

def rr_define(sig, sampling_rate=256):
    # peak_indices = np.argwhere(sig == np.amax(sig))
    
    rr_intervals = []
    peak_value = np.amax(sig)
    current_interval = 0

    for v in sig:
        if v == peak_value:
            if current_interval <= 100:
                # print("SAME" + str(len(rr_intervals)))
                current_interval += 1
                continue
            rr_intervals.append(current_interval)
            current_interval = 0
        else:
            current_interval += 1
    rr_intervals = np.array(rr_intervals)/sampling_rate
    return rr_intervals

def plot(sig, title = "Plot of CT signal", sampling_rate = 256):
    end_time = sig.shape[0]-1/sampling_rate
    t = np.linspace(0, end_time , sig.shape[0])
    plt.plot(t, sig)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title(title)
    plt.xlim([0, end_time])
    plt.show()


if __name__ == '__main__':
    raw_signal = pd.read_csv("DataN.txt", header=None)[0][:2000]
    rr_graph = qrs_detect(raw_signal, win_size = 25)
        
    
    



