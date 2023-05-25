from src import np, pd, sp, skew, kurtosis

def extract_features(data, feature_names):
    
    """
    Extract a set of time domain features given one window
    :param data: data from one window
    :param features_names: list of all features to calcultate
    """
    # Work with numpy matrix
    data = np.array(data)

    var = np.var(data) #Variance(VAR)
    rms = np.sqrt(np.mean(data ** 2)) #RootMeanSquare(RMS)
    mav = np.sum(np.absolute(data)) / len(data) #MeanAbsoluteValue (MAV)
    wl = np.sum(abs(np.diff(data))) # Waveform Length (WL)
    mean = np.mean(data) # Mean(MEAN)
    std = np.std(data) # StandardDeviation(SD)
    median = np.median(data) # Median
    peak = np.max(data) #Peak(PK)
    min = np.min(data) # Min(MIN)
    iemg = np.sum(abs(data)) # IntegralEMG(iEMG)
    aac = np.sum(abs(np.diff(data))) / len(data) # AverageAmplitudeChange(ACC)
    kur = kurtosis(data) # Kurtosis(KURT)
    skewe = skew(data) # Skewness(SKEW)

    matrix = np.column_stack((var, mav, rms, wl, mean, std, median, peak, min, iemg, aac, kur, skewe))
    df_features = pd.DataFrame(matrix, columns=feature_names)
    
    return df_features
