from combine_data import combine_data
import librosa
from math import floor
import numpy as np
import pandas as pd

def _split_columns(df:pd.DataFrame,column_name:str,length:int) ->pd.DataFrame:
    '''
    a simple function that split the column of list to multiple columns
    and return a data frame object
    '''
    column_names = [column_name + str(i) for i in range(1,length+2)]
    #, columns=column_names
    return pd.DataFrame(df[column_name].to_list(),columns=column_names)




def frame_v1(y,no_frames,overlapping_percetage = 0.25):
    """            
    frame function
    based on the librosa framing function it returns a matrix with overlapping frames from the samples.
    it takes a sample as an np.array of floats.
    no_frames = integer of frames
    overlapping percentage = the overlapping percentage between the frames -- need to be updated for the moment.
    returns a 2D matrix with the shape (samples in frames,frames)
     -- this is an example from librosa
    [0, 1, 2, 3, 4, 5, 6] for frame length of 3 and hop length of 2
         ||
        \\//
     [[0, 2, 4],
      [1, 3, 5],
      [2, 4, 6]]

    """
    no_frames = int(len(data)/no_frames) # calculating the number of samples in each frame, from the number of frames.

    # calculating the hop length from the overlapping percentage -need editing.
    hop_length = int((1-overlapping_percetage)*len(data)/no_frames) 
    return librosa.util.frame(y,frame_length=no_frames,hop_length = hop_length)


def framer_v2(y,frame_length,hop_length):
    """            
    frame function
    based on the librosa framing function it returns a matrix with overlapping frames from the samples.
    it takes a sample as an np.array of floats.
    frames length = number of samples in the frame.
    hop_length = samples to skip to the next frame.
    returns a 2D matrix with the shape (samples in frames,frames)
     -- this is an example from librosa
    [0, 1, 2, 3, 4, 5, 6] for frame length of 3 and hop length of 2
         ||
        \\//
     [[0, 2, 4],
      [1, 3, 5],
      [2, 4, 6]]

    """
    return librosa.util.frame(y,frame_length=frame_length,hop_length = hop_length)


### redefining features for the feature extraction function.###
def mean(y,frame_length,hop_length):
    return framer_v2(y,frame_length,hop_length).mean(axis=0)

def variance(y,frame_length,hop_length):
    return framer_v2(y,frame_length,hop_length).var(axis=0)

def zero_crossing_rate(y,frame_length,hop_length):
    frames = framer_v2(y,frame_length,hop_length)
    signs = np.sign(frames)
    difference = np.diff(signs,axis=0)
    absolute = np.abs(difference)
    means= np.mean(absolute,axis=0)
    return means / 2

def iemg(y,frame_length,hop_length):
    return np.abs(framer_v2(y, frame_length, hop_length)).sum(axis=0)

def extract_features(data,label,features_no,overlapping_percentage=0.25,features_funcs=[]):
    """Extracts features from data and returns the fitted model object.
    create,recieves the processed and returns the labeled dataset for training the model.
    data : processed data np.array
    labels : lable for the data np.array
    features_no : number of measurements from eac feature
    feature_funcs : list or iterable feature extraction functions
    the feature function should accept the data as the first argument, hop_length for the second argumet,
    
    """
    features = {}
    column_names = []
    for func in features_funcs:
        column_names.append(func.__name__)
        features[func.__name__] = []
        for samples in data:
            features[func.__name__].append(func(y = samples,frame_length=floor(len(data)/features_no-1),hop_length=int((1-overlapping_percentage)*floor((len(samples))/features_no))))
    features['labels'] = label

    df = pd.DataFrame(features)
    print(df)
    labels = df.labels
    df = pd.concat([_split_columns(df,column_name=column,length=features_no) for column in column_names],axis=1)
    df['label'] = labels
    return df

if __name__ == '__main__':
    data,label,_= combine_data('./../data') # some tests.
    print(extract_features(data = data,label = label,features_no= 3,overlapping_percentage=0.25,features_funcs=[mean,zero_crossing_rate,iemg]))