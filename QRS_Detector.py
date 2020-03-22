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
    - N (win_size) should be set approximately the same as the widest possible QRS
complex
"""

def qrs_detect(raw_signal, win_size=15, save=False,
               enable_noise_filter=True, plotlimit=2000):
    """

    Parameters
    ----------
    raw_signal : TYPE

    win_size : TYPE
        the moving average window size N.

    Returns
    -------
    a vector that contains the timestamps of the R wave and a vector that
    contains the corresponding RR intervals

    """
    raw_signal = np.array(raw_signal)
    if enable_noise_filter:
        noise_filtered_signal = remove_noise(raw_signal)
        
        plot_before_after(raw_signal, noise_filtered_signal,
                          title = "Before_After_Filter",
                          save=save, limit=plotlimit)
    
        diff = differentiate(noise_filtered_signal)
    else:
        diff = differentiate(raw_signal)
        
    plot(diff, title="Differentiated Signal", limit=plotlimit)

    sqrd = square(diff)
    plot(sqrd, title="Squared Signal", limit=plotlimit)

    smoothed = smooth(sqrd, win_size)
    plot(smoothed, title="Smoothed Signal", limit=plotlimit)

    thresholded, threshold1 = threshold(smoothed)
    plot(thresholded, title="Thresholded Signal", limit=plotlimit)

    rr_intervals = rr_define(thresholded)
    plot(rr_intervals*1000, title="RR", sampling_rate=1,
         xlabel='Beat Number', ylabel='time(ms)',
         limit=rr_intervals.shape[0], save=save)
    
    r_markers = create_markers(rr_define(thresholded[:plotlimit]))
    final_figure_title = "DetectedR_" + str(win_size)
    if not enable_noise_filter:
        final_figure_title = "Unfiltered_" + str(win_size)
    plot(thresholded, title = final_figure_title , sampling_rate=256,
         xlabel="t", ylabel="x(t)", markers=r_markers, save=save,
         limit=plotlimit)

    return rr_intervals

def create_markers(rr_intervals, sampling_rate=256):

    r_markers = rr_intervals*256
    first_might_miss_beat = np.argmax(r_markers)
    if first_might_miss_beat not in r_markers:
        tmp = np.zeros((r_markers.shape[0]+1,))
        tmp[0] = first_might_miss_beat
        tmp[1:] = r_markers
        r_markers = tmp
    for i in range(1, len(r_markers)):
        r_markers[i] = r_markers[i] + r_markers[i-1]
    return r_markers

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

def threshold(sig, avg_mult=1.5):
    threshold = np.average(sig)*avg_mult
    # threshold = np.max(sig)*0.7
    thresholded = sig.copy()
    thresholded[thresholded > threshold] = threshold
    return thresholded, threshold

def rr_define(sig, sampling_rate=256, discard_less_than=100):
    # peak_indices = np.argwhere(sig == np.amax(sig))
    
    rr_intervals = []
    peak_value = np.amax(sig)
    current_interval = 0

    for v in sig:
        if v == peak_value:
            if current_interval <= discard_less_than:
                # print("SAME" + str(len(rr_intervals)))
                current_interval += 1
                continue
            rr_intervals.append(current_interval)
            current_interval = 0
        else:
            current_interval += 1
    rr_intervals = np.array(rr_intervals)/sampling_rate
    return rr_intervals

def plot(sig, title = "Plot of CT signal", sampling_rate=256,
         xlabel="t", ylabel="x(t)", markers=None, save=False,
         limit=2000):
    draw_sig=sig[:limit]
    end_time = draw_sig.shape[0]-1/sampling_rate
    t = np.linspace(0, end_time , draw_sig.shape[0])
    if markers is not None:
        markers_amp = np.ones(markers.shape)*np.max(draw_sig)
        plt.plot(markers, markers_amp, marker="*",
                 linestyle=' ', color='r', label='R-Waves')
    plt.plot(t, draw_sig)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, end_time])  
    if save:
        plt.savefig(title + ".jpg", progressive=True)
    plt.show()
        
        
def plot_before_after(before_sig, after_sig,
                      title = "Before & After of CT signal",
                      sampling_rate=256,
                      save=False, limit=2000):
    draw_before_sig = before_sig[:limit]
    draw_after_sig = after_sig[:limit]
    end_time = draw_before_sig.shape[0]-1/sampling_rate
    t = np.linspace(0, end_time , draw_before_sig.shape[0])
    plt.subplot(211)
    plt.plot(t, draw_before_sig)
    plt.subplot(212)
    plt.plot(t, draw_after_sig)
    if save:
        plt.savefig(title + ".jpg", progressive=True)
    plt.show()

if __name__ == '__main__':
    raw_signal = pd.read_csv("DataN.txt", header=None)[0][:]
    rr_graph = qrs_detect(raw_signal, win_size = 25,
                          save=True, enable_noise_filter=False,
                          plotlimit=2000)
    rr_graph = qrs_detect(raw_signal, win_size = 5,
                          save=True, enable_noise_filter=True,
                          plotlimit=2000)
    rr_graph = qrs_detect(raw_signal, win_size = 15,
                          save=True, enable_noise_filter=True,
                          plotlimit=2000)
    rr_graph = qrs_detect(raw_signal, win_size = 25,
                          save=True, enable_noise_filter=True,
                          plotlimit=2000)
        
    
    



