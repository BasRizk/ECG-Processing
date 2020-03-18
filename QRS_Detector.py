# -*- coding: utf-8 -*-
"""
Apply your function to the ECG signal provided in the file “DataN.txt”.
The sampling rate of this ECG signal is 256 Hz. You will need to suggest a
method to compute the threshold needed for detection.
"""

def qrs_detect(raw_signal, win_size):
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
    pass