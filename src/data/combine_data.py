from src import os, pd, np

def combine_data(path: str):
    """
    this script will read the json files and structures it itno X,y for preprocessing and feature extraction.
    the input is the son data path.
    this function will outputs 3 np.array tuples raw data, labels, timestamps.
    """
    data = []
    labels = []
    time_stamps = []
    for file_name in os.listdir(path):
        if file_name.endswith("json"):
            df = pd.read_json(path + '/' + file_name)
            data.append(np.asarray(df.data.values, dtype=np.float32))
            labels.append(df.label.unique()[0])
            time_stamps.append(df.time)
    data = np.array(data)
    labels = np.array(labels)
    time_stamps = np.array(time_stamps)
    return data, labels, time_stamps
