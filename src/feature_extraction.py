from combine_data import combine_data
from librosa.feature import zero_crossing_rate
import numpy as np
import pandas as pd



def frame_v1(data,no_frames,overlapping_percetage = 0.25):
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
    return librosa.util.frame(data,frame_length=no_frames,hop_length = hop_length)
def framer_v2(data,frame_length,hop_length):
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
    return librosa.util.frame(data,frame_length=frame_length,hop_length = hop_length)


def extract_features(data,label,features_no,*features_funcs,):
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

            features[func.__name__].append(func(y = samples,hop_length=1+int((len(samples))/features_no))[0])
    features['labels'] = label
    df = pd.DataFrame(features)
    for func in column_names:
        df[[name+str(j) for name in column_names for j in range(1,features_no+1)] ] = pd.DataFrame(df[func].tolist(), index= df.index)
        df.drop(columns=column_names,inplace=True)
    return df

if __name__ == '__main__':
    data,label,_= combine_data('./../data') # some tests.
    print(extract_features(data,label,4,zero_crossing_rate,))