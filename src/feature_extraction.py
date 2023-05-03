import math

from src import np, pd

def extract_features(data, overlap):
    """
    Extracts features from data and returns the fitted model object.
    :param data: data with signals
    :param overlap: overlaping space
    """
    # Work with numpy matrix
    data_n = data.to_numpy()
       
    # Time domain features: Mean absolute value (MAV), root mean square (RMS), waveform length (WL),zero crossings (ZC), variance (VAR)
    feature_names = ['Variance(VAR)', 'MeanAbsoluteValue(MAV)', 'RootMeanSquare(RMS)', 'WaveformLength(WL)']
    var = []
    mav = []
    rms = []
    wl = []
    zc = []

    # Iterate for each signal/window and calculate associated features
    for x in range(len(data_n)):
  
        var.append(np.var(data_n[x]))
        rms.append(np.sqrt(np.mean(data_n[x] ** 2)))
        mav.append(np.sum(np.absolute(data_n[x])) / len(data_n[x]))
        wl.append(np.sum(abs(np.diff(data_n[x]))))
        #zc.append(zcruce(x, th))
        
    matrix = np.column_stack((var, mav, rms, wl))
    df_features = pd.DataFrame(matrix, columns = feature_names)
    data = pd.concat([data, df_features], axis=1)

    return data
