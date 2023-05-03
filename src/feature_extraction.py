from combine_data import combine_data
from librosa.feature import zero_crossing_rate
import numpy as np
import pandas as pd
data,label,_= combine_data('./../data') # some tests.
def extract_features(data,label,features_no,*features_funcs,):
    """Extracts features from data and returns the fitted model object.
    create,recieves the processed and returns the labeled dataset for training the model.
    data : processed data np.array
    labels : lable for the data np.array
    features_no : number of measurements from eac feature
    feature_funcs : list or iterable feature extraction functions
    the feature function should accept the data as the first argument, hop_length for the second argumet,
    
    """
    print(type(data))
    features = {}
    for func in features_funcs:
        features[func.__name__] = []
        for i,samples in enumerate(data):
            features[func.__name__].append(func(y = samples,hop_length=1+int((len(samples))/features_no))[0])
    features['labels'] = label
    return pd.DataFrame(features)

if __name__ == '__main__':
    print(extract_features(data,label,3,zero_crossing_rate))