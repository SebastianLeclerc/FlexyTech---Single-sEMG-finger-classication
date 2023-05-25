import os
import pandas as pd
import numpy as np
import re


def combine_data(path: str):
    """
    this script will read the json files and structures it itno X,y for preprocessing and feature extraction.
    the input is the son data path.
    this function will outputs 2 np.array tuples raw data, labels
    """
    data = []
    labels = []
    
    # Iterate for each sample data json file in the directory
    for file_name in os.listdir(path):
        if file_name.endswith("json"):
            df = pd.read_json(path + '/'+file_name)
            data.append(np.asarray(df.data.values, dtype=np.float32))
            # Define labels based on file_name - e.g Original file name = 1IndexFlexion -> label = IndexFlexion
            labels.append(re.findall(pattern=r'[A-Z]\w+', string=file_name)[0])
    
    # Make sure output data type is np.array
    data = np.array(data)
    labels = np.array(labels)

    return data, labels
