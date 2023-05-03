import math

from src import np, pd
def extract_features(data, overlap):
    """
    Extracts features from data and returns the fitted model object.
    :param data: data with signals
    :param overlap: overlaping space
    """

    # Time domain features
    features_names = ['Variance(VAR)', 'MeanAbsoluteValue(MAV)', 'RootMeanSquare(RMS)', 'WaveformLength(WL)', 'ZeroCrossings(ZC)']
    var = []
    mav = []
    rms = []
    wl = []
    #zc = []

    # Iterate for each signal/window and calculate associated features
    for index in range(len(data)):
        var.append(data.iloc[[index]].var(axis=1))
        mav.append(sum(abs(data.iloc[[index]].var(axis=1)))/len(data.iloc[[index]]))
        rms.append(math.sqrt((data.iloc[[index]].mean(axis=1))**2))
        wl.append(sum(abs(data.iloc[[index]].diff(axis=1)))
        #zc.append(data.iloc[[index]].mean(axis=1))
    
    var = np.array(var)
    mav = np.array(mav)
    rms = np.array(rms)
    wl = np.array(wl)
    #zc = np.array(zc)

    matrix = np.column_stack((mav, rms, wl, zc))
    df_features = pd.DataFrame(matrix, columns = features_names)
    data = pd.concat([data, df_features], axis=1)

    return data
