from src import np, pd
import scipy as sp
from scipy.stats import skew, kurtosis

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
