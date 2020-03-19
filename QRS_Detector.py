# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
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
    
    noise_filtered_signal = remove_noise(raw_signal)
    diff = differentiate(noise_filtered_signal)
    sqrd = square(diff)
    smoothed = smooth(sqrd)
    thresholded = threshold(smoothed)
    rr_intervals = rr_define(thresholded)
    return rr_intervals

def remove_noise(signal):
    signal = notch_filter(signal)
    signal = bandpass_filter(signal)
    return signal

def notch_filter(signal):
    #TODO
    pass

def bandpass_filter(signal):
    #TODO
    pass

def differentiate(signal):
    # TODO
    pass

def square(signal):
    # TODO
    pass

def smooth(signal):
    # TODO
    # smooth the squared signal using a moving average window
    
    pass

def threshold(signal):
    # TODO
    pass

def rr_define(rr_intervals):
    # TODO
    pass



if __name__ == '__main__':
    raw_signal = pd.read_csv("DataN.txt", header=None)
    rr_graph = qrs_detect(raw_signal) 
    
    
    


