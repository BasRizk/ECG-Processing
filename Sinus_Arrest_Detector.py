# -*- coding: utf-8 -*-
import QRS_Detector
import pandas as pd
import numpy as np

def discoverMissingBeats(sig, win_size = 25) :
    noise_free_signal = QRS_Detector.remove_noise(sig)
    diff_signal = QRS_Detector.differentiate(noise_free_signal)
    sqrd_signal = QRS_Detector.square(diff_signal)
    smoothed_signal = QRS_Detector.smooth(sqrd_signal, win_size)
    thresholded_signal, P_threshold = QRS_Detector.threshold(smoothed_signal)
    rr_intervals = QRS_Detector.rr_define(thresholded_signal)
    average_time_between_P_waves = np.average(rr_intervals)
    missing_beats_timestamps = calculateMissingTimes(thresholded_signal, P_threshold, average_time_between_P_waves)
    print(missing_beats_timestamps)
    QRS_Detector.plot(smoothed_signal, "Smoothed sinus arrest")
    QRS_Detector.plot(thresholded_signal, "Theresholded Signal sinus arrest")
    QRS_Detector.plot(rr_intervals, "RR_Intervals sinus arrest", sampling_rate=1)
    return missing_beats_timestamps
    
def calculateMissingTimes(sig, threshold, average_time):
    timestamps_array = []
    current_timestamp_between_P = 0
    for i in range(len(sig)) :
        if(sig[i] < threshold):
            current_timestamp_between_P += 1/256
        elif(current_timestamp_between_P - average_time > 0.5 and current_timestamp_between_P > average_time) :
            normalized_i = i * (1/256)
            timestamps_array.append((int)(256*((normalized_i-current_timestamp_between_P)+current_timestamp_between_P*0.5)))
            current_timestamp_between_P = 0
        else :
            current_timestamp_between_P = 0
            
    return timestamps_array

if __name__ == '__main__':
    raw_signal = pd.read_csv("Data2.txt", header=None)[0][:]
    missing_beats_timestamps = discoverMissingBeats(raw_signal)
    to_be_written_to_file = ""
    file = open('MissingBeats.txt', 'w')
    to_be_written_to_file = to_be_written_to_file + "There is " + str(len(missing_beats_timestamps)) + " beats\n"
    to_be_written_to_file = to_be_written_to_file + "They should have been present in the following " + str(len(missing_beats_timestamps)) + " samples\n"
    for i in range(len(missing_beats_timestamps)) :
        to_be_written_to_file = to_be_written_to_file + str(missing_beats_timestamps[i]) + "\n"
    file.write(to_be_written_to_file)
    file.close()
    