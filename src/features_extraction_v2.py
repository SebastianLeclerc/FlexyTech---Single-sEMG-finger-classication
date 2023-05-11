from src import np, pd
from scipy.stats import skew, kurtosis
from src.data.combine_data import combine_data
import librosa
from math import floor

def extract_features(data, overlap):
    """
    Extracts features from data and returns the fitted model object.
    :param data: data with signals
    :param overlap: overlaping space
    """
    # Work with numpy matrix
    data_n = data.to_numpy()
       
    # Time domain features: Mean absolute value (MAV), root mean square (RMS), waveform length (WL),zero crossings (ZC), variance (VAR)
    feature_names = ['Variance(VAR)', 'MeanAbsoluteValue(MAV)', 'RootMeanSquare(RMS)', 'WaveformLength(WL)', 'Mean(MEAN)', 
                     'StandardDeviation(SD)', 'Median', 'Peak(PK)', 'Min(MIN)', 'IntegralEMG(iEMG)', 'AverageAmplitudeChange(ACC)',
                     'Kurtosis(KURT)', 'Skewness(SKEW)']
    var = []
    mav = []
    rms = []
    wl = []
    zc = []
    mean = []
    std = []
    median = []
    peak = []
    min = []
    iemg = []
    aac = []
    kur = []
    skewe = []

    # Iterate for each signal/window and calculate associated features
    for x in range(len(data_n)):
  
        var.append(np.var(data_n[x]))
        rms.append(np.sqrt(np.mean(data_n[x] ** 2)))
        mav.append(np.sum(np.absolute(data_n[x])) / len(data_n[x]))
        wl.append(np.sum(abs(np.diff(data_n[x]))))
        mean.append(np.mean(data_n[x]))
        std.append(np.std(data_n[x]))
        median.append(np.median(data_n[x]))
        peak.append(np.max(data_n[x]))
        min.append(np.min(data_n[x]))
        iemg.append(np.sum(abs(data_n[x])))
        aac.append(np.sum(abs(np.diff(data_n[x]))) / len(data_n[x]))
        kur_value = kurtosis(data_n[x])
        kur.append(kur_value)
        ske_value = skew(data_n[x])
        skewe.append(ske_value)


        #zc.append(zcruce(x, th))

    matrix = np.column_stack((var, mav, rms, wl, mean, std, median, peak, min, iemg, aac, kur, skewe))
    df_features = pd.DataFrame(matrix, columns = feature_names)
    data = pd.concat([data, df_features], axis=1)

    return data

def _split_columns(df:pd.DataFrame,column_name:str,length:int) ->pd.DataFrame:
    '''
    a simple function that split the column of list to multiple columns
    and return a data frame object
    '''
    column_names = [column_name + str(i) for i in range(1,length+2)]
    #, columns=column_names
    return pd.DataFrame(df[column_name].to_list(),columns=column_names)




def _frame_v1(y,no_frames,overlapping_percetage = 0.25):
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


def _framer_v2(y,frame_length,hop_length):
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
    return _framer_v2(y,frame_length,hop_length).mean(axis=0)

def variance(y,frame_length,hop_length):
    return _framer_v2(y,frame_length,hop_length).var(axis=0)

def zero_crossing_rate(y,frame_length,hop_length):
    frames = _framer_v2(y,frame_length,hop_length)
    signs = np.sign(frames)
    difference = np.diff(signs,axis=0)
    absolute = np.abs(difference)
    means= np.mean(absolute,axis=0)
    return means / 2

def iemg(y,frame_length,hop_length):
    return np.abs(_framer_v2(y, frame_length, hop_length)).sum(axis=0)

def root_mean_squared(y,frame_length,hop_length):
    return np.sqrt(np.mean(_framer_v2(y, frame_length, hop_length) ** 2,axis=0))

def mean_absolute_value(y,frame_length,hop_length):
    return np.mean(np.abs(_framer_v2(y, frame_length, hop_length)),axis=0)

def wave_form_length(y,frame_length,hop_length):
    frames = _framer_v2(y, frame_length, hop_length)
    diff = np.diff(frames,axis=0)
    absolute = np.abs(diff)
    sum = np.sum(absolute,axis=0)
    return sum

def standard_deviation(y,frame_length,hop_length):
    frames = _framer_v2(y, frame_length, hop_length)
    return np.std(frames,axis=0)

def median(y,frame_length,hop_length):
    frames = _framer_v2(y,frame_length,hop_length)
    return np.median(frames,axis=0)

def min(y,frame_length,hop_length):
    frames = _framer_v2(y, frame_length, hop_length)
    return np.min(frames,axis=0)


def max(y,frame_length,hop_length):
    frames = _framer_v2(y, frame_length, hop_length)
    return np.max(frames,axis=0)

def skewness(y,frame_length,hop_length):
    frames = _framer_v2(y, frame_length, hop_length)
    return skew(frames, axis=0)

def kurt(y,frame_length,hop_length):
    frames = _framer_v2(y, frame_length, hop_length)
    return kurtosis(frames, axis=0)




#### to do #########
# 'MeanAbsoluteValue(MAV)' -- done not tested.
# 'WaveformLength(WL)'     -- done not tested.
# StandardDeviation(SD)'  -- done not tested.
#  'Median'               -- done not tested.
# 'Peak(PK)'
#  'Min(MIN)'            --done not tested.
# 'AverageAmplitudeChange(ACC)'
# 'Kurtosis(KURT)'       --done not tested.
# 'Skewness(SKEW)'        --done not tested.



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
    print(extract_features(data = data,label = label,features_no= 3,overlapping_percentage=0.25,features_funcs=[mean,root_mean_squared,iemg]))
