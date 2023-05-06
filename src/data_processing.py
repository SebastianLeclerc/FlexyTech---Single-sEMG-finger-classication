import pandas as pd
import os
from combine_data import combine_data  #from the module 'combine_data' import the function 'combine_data'
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def filter(voltage_values,low_cut=1,high_cut=55,fs = 1e3,):
    '''
    band pass filter based on the scipy butter bandpass, allowing frequency between low cut and high cut to pass through.
    input : np.array of the data voltage values.
    low_cut : lowest frequency defaults at 1
    high cut : highest frequency in the band filter defaults at 55.
    '''
    return butter_bandpass_filter(voltage_values,low_cut=low_cut,highcut=high_cut,fs = fs)


def ADC_to_v(sample:np.array,max_voltage = 1.8,ADC_resolution = 4095,resting_voltage=1.5):
    return sample * (max_voltage)/ADC_resolution - 1.5




def normalize_data(sample:np.array):
    """
    normalize the the data in the samples between -1,1.
    sample: the sample as an np.array.
    returns the  normalized sample as an np.array.
    """
    max = np.max(sample)                                    #Saves the maximum value of each data file
    min = np.min(sample)                                    #Saves the minimum value of each data file
    diff = max - min                                            #Saves the difference
    min_vector = np.ones(len(sample)) * min                 #Creates a vector of only 1 with the correct length and multiplies it with the minimum value
    return (2 * (sample - min_vector)/diff)-1          #Normalize the entire vector within the range of [-1,1]

def format_timestamps(time_stamps:np.array):
    '''
    recenter the timestamps.
    time_stamps : np.array of time stamps of the data.
    retrun centered time stamps.
    '''
    time_vector = np.ones(len(time_stamps)) * time_stamps[0]  #Creates a vector of only 1 with the correct length and multiplies it with the first time stamp
    return time_stamps-time_vector                 #Subtracts the entire time stamp vector with the first value in order to see the duration of the data set


def preprocess_data(filepath):
    """
    Loads data from file and returns processed data.
    filepath: directory of the data.
    returns a tuple of the processed data, labels, and formated timestamps.
    
    
    """
    #df = pd.read_csv('./../data', delimiter=r"\s+")
    data,labels,time_stamps = combine_data(filepath) #call the function combine_data with the correct file path. ./../data -> from current directory / go back / go to 'data'
    # To Do
    # scale the data and return the scaled data
    for i in range(len(data)):
        data[i]= normalize_data(sample = data[i])
    
    for i in range(len(data)):
        time_stamps[i] = format_timestamps(time_stamps[i])

    return data, labels, time_stamps


if __name__ == '__main__':
    print(preprocess_data('./../data'))
#print(preprocess_data('./../data'))