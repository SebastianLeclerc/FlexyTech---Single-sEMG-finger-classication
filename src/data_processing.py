import pandas as pd
import os
from combine_data import combine_data  #from the module 'combine_data' import the function 'combine_data'
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def preprocess_data(filepath):
    """Loads data from file and returns processed data."""
    #df = pd.read_csv('./../data', delimiter=r"\s+")
    data,labels,time_stamps = combine_data('./../data') #call the function combine_data with the correct file path. ./../data -> from current directory / go back / go to 'data'
    # To Do
    # scale the data and return the scaled data
    for i in range(len(data)):
        max = np.max(data[i][0])                                    #Saves the maximum value of each data file
        min = np.min(data[i][0])                                    #Saves the minimum value of each data file
        diff = max - min                                            #Saves the difference
        min_vector = np.ones(len(data[i][0])) * min                 #Creates a vector of only 1 with the correct length and multiplies it with the minimum value
        data[i][0]= (2 * (data[i][0] - min_vector)/diff)-1          #Normalize the entire vector within the range of [-1,1]
    
    for i in range(len(data)):
        time_vector = np.ones(len(data[i][0])) * time_stamps[i][0]  #Creates a vector of only 1 with the correct length and multiplies it with the first time stamp
        time_stamps[i] = time_stamps[i]-time_vector                 #Subtracts the entire time stamp vector with the first value in order to see the duration of the data set

        

    # rescale by using formula -> [a,b] = (b-a)((x-min(data))/(max(data)-min(data)))) where a<b.
    
    #! Substract the first value from all the other values in the time_stamps array 

    return data, labels, time_stamps

preprocess_data('./../data')
#print(preprocess_data('./../data'))