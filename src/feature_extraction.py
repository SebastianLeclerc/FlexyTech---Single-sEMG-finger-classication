from combine_data import combine_data
from librosa.feature import zero_crossing_rate
import numpy as np
import pandas as pd
data,label,_= combine_data('./../data') # some tests.



def extract_features(data,label,features_no,*features_funcs,frame_percentage_of_the_sample = 0.5):
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
        for samples in enumerate(data):
            features[func.__name__].append(func(y = samples,frame_length = int(len(samples)*frame_percentage_of_the_sample),hop_length=1+int((len(samples))/features_no))[0])
    features['labels'] = label
    df = pd.DataFrame(features)
    for func in column_names:
        df[[name+str(j) for name in column_names for j in range(1,features_no+1)] ] = pd.DataFrame(df[func].tolist(), index= df.index)
        df.drop(columns=column_names,inplace=True)
    return df

if __name__ == '__main__':
    print(extract_features(data,label,4,zero_crossing_rate,))