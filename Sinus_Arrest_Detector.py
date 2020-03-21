# -*- coding: utf-8 -*-
import QRS_Detector
import pandas as pd

def discoverMissingBeats(sig, win_size = 25) :
    noise_free_signal = QRS_Detector.remove_noise(sig)
    diff_signal = QRS_Detector.differentiate(noise_free_signal)
    sqrd_signal = QRS_Detector.square(diff_signal)
    smoothed_signal = QRS_Detector.smooth(sqrd_signal)
    theresholded_signal = QRS_Detector.threshold(smoothed_signal)
    rr_intervals = QRS_Detector.rr_define(theresholded_signal)
    QRS_Detector.plot(smoothed_signal, "Smoothed sinus arrest")
    QRS_Detector.plot(rr_intervals, "RR_Intervals sinus arrest", sampling_rate=1)
    
if __name__ == '__main__':
    raw_signal = pd.read_csv("Data2.txt", header=None)[0][:2000]
    discoverMissingBeats(raw_signal)
