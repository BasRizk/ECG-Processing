# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    plot(raw_signal, title="Raw Signal")
    
    diff = differentiate(raw_signal)
    sqrd = square(diff)
    smoothed = smooth(sqrd)
    thresholded = threshold(smoothed)
    rr_intervals = rr_define(thresholded)
    return rr_intervals

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

def plot(signal, title = "Plot of CT signal"):
    t = np.linspace(-0.02, 0.05, signal[0].shape[0])
    plt.plot(t, signal[0])
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title(title)
    plt.xlim([-0.02, 0.05])
    plt.show()

if __name__ == '__main__':
    raw_signal = pd.read_csv("DataN.txt", header=None)
    rr_graph = qrs_detect(raw_signal) 
        
    
    


