import os
import pickle
import json
import re
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import skew, kurtosis
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import *
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#Read all data
def combine_data(path:str):
    """
    this script will read the json files and structures it itno X,y for preprocessing and feature extraction.
    the input is the son data path.
    this function will outputs 3 np.array tuples raw data, labels, timestamps.
    """
    data = []
    labels = []
    for file_name in os.listdir(path):
        if file_name.endswith("json"):
            df = pd.read_json(path + '/'+file_name)
            data.append(np.asarray(df.data.values,dtype=np.float32))
            #labels.append(df.label.unique()[0])
            labels.append(re.findall(pattern=r'[A-Z]\w+',string=file_name)[0])
            #time_stamps.append(df.time)
    data = np.array(data)
    labels = np.array(labels)
    #time_stamps = np.array(time_stamps)
    return data,labels


#Create the windows
def sliding_window(data, window_size, overlap):
    """
    Applies a sliding window with overlap to the input data.

    Parameters:
    data (ndarray): input data
    window_size (int): size of sliding window
    overlap (int): overlap between consecutive windows

    Returns:
    ndarray: 2D array with sliding windows of shape (n_windows, window_size)
    """
    n_samples = data.shape[0]
    overlap = int(np.floor((window_size*overlap)/100))
    n_windows = int(np.floor((n_samples - window_size) / overlap) + 1)
    windows = np.zeros((n_windows, window_size))

    for i in range(n_windows):
        start = i * overlap
        end = start + window_size
        windows[i] = data[start:end]

    return windows
    

    

def feature_windows(data,window_size =250,overlap = 50):
    '''
    a function that measures features from each slides
    and creates a 1D array of measurements from each slide.
    '''
    var    = lambda data : np.var(data,axis =1) 
    rms    = lambda data : np.sqrt(np.mean(data ** 2,axis =1))
    mav    = lambda data : np.sum(np.absolute(data),axis =1) / len(data)
    wl     = lambda data : np.sum(abs(np.diff(data)),axis =1)
    mean   = lambda data : np.mean(data,axis =1 )
    std    = lambda data : np.std(data,axis =1)
    median = lambda data : np.median(data,axis =1)
    peak   = lambda data : np.max(data,axis =1)
    min    = lambda data : np.min(data,axis =1)
    iemg   = lambda data : np.sum(abs(data),axis =1)
    aac    = lambda data : np.sum(abs(np.diff(data)),axis =1) / len(data)
    kur    = lambda data : kurtosis(data,axis =1)
    skewe  = lambda data : skew(data,axis =1)
    features = [var,rms,mav,wl,mean,std,median,peak,min,iemg,aac,kur,skewe]
    win_matrix = sliding_window(data, window_size, overlap)
    feature_vector = []
    for feature_func in features:
        feature_vector.extend(feature_func(win_matrix))
    return np.array(feature_vector)
          


# reads the data from JSON files and stores it in memory
data,labels = combine_data(r'C:\Users\Nazem\OneDrive\Desktop\samples200\samples')

# segments the data samples into several overlapping views and calculates some measurements, from each view.           
processed_data = np.array([feature_windows(i) for i in data])

# including the classes in a inclusion criteria
inclusion_criteria_flexion = ['IndexFlexion', 'MiddleFlexion','RingFlexion','LittleFlexion','ThumbFlexion']
inclusion_criteria_extension = ['IndexExtension', 'MiddleExtension','RingExtension','LittleExtension','ThumbExtension']
inclusion_criteria_3_ = ['IndexFlexion', 'MiddleFlexion','RingFlexion','IndexExtension', 'MiddleExtension','RingExtension']
inclusion_criteria_all = inclusion_criteria_extension + inclusion_criteria_flexion
inclusion_criteria_3_flexion = ['IndexFlexion', 'MiddleFlexion','RingFlexion']

# locating the indices for the classes chosen
inclusion_idx = np.isin(labels, inclusion_criteria_3_flexion )

X = processed_data[inclusion_idx]
y =labels[inclusion_idx]

# Split data in training and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = 'rf' # the name of the model to train the data
# pca
pca = PCA()
# Define a Standard Scaler to normalize inputs
scaler = RobustScaler()
scaler_MinMax = MinMaxScaler
# defining the estimators
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
svm = SVC()
mlp = MLPClassifier()


# pipelines definitions for multiple models
# each pipeline entry has a unique name,
# a list of tuples indicating the name of the step, and a reference to the class
# parameters dictionary if any parameters is passed.

Pipelines = {
    'knn':{
        'name':'knn_scaled',
        'steps':[('scaler',scaler),('knn',knn)],
        'params':{
            'knn__leaf_size': [20], 
            'knn__metric': ['minkowski'], 
            'knn__n_neighbors': [1],
            'knn__p': [1], 
            'knn__weights': ['uniform']
        }

    },
    'rf':{
        'name':'rf',
        'steps':[('rf',rf)],
        'params':{
            'rf__criterion': ['gini'], 
            'rf__max_depth': [6], 
            'rf__max_features': ['sqrt'], 
            'rf__n_estimators':[ 500]
        }
    },
    'gb':{
        'name':'gb',
        'steps':[('gb',gb)],
        'params':{
            'gb__criterion': ['friedman_mse'], 
            'gb__learning_rate': [0.1], 
            'gb__loss': ['deviance'], 
            'gb__max_features': ['sqrt'], 
            'gb__n_estimators': [15]
        }
    },
    'svc_RobustScaler_pca':{
        'name':'svc_RobustScaler_pca',
        'steps': [('scaler',scaler),('pca',pca),('svc',svm)],
        'params':{
            'pca__n_components': [10, 20, 30, 40], 
            'svc__C': [1, 10, 100], 
            'svc__gamma': [0.001, 0.01, 0.1, 01], 
            'svc__kernel': ['rbf']
        }
        
    },
    'svc_MinMaxScaler_pca':{
        'name':'svc_MinMaxScaler_pca',
        'steps': [('scaler',scaler_MinMax),('pca',pca),('svc',svm)],
        'params':{
            'pca__n_components': [10, 20, 30, 40],
            'svc__C': [1, 10, 100],
            'svc__gamma': [0.001, 0.01, 0.1, 01], 
            'svc__kernel': ['rbf']
        }
    },
    'mlp_RobustScaler_pca':{
        'name':'mlp_RobustScaler_pca',
        'steps': [('scaler',scaler),('pca',pca), ('mlp',mlp)],
        'params':{
            'pca__n_components': [10, 20, 30, 40],
            'mlp__hidden_layer_sizes': [(120,60),(200,50,300,40), (300,10,200,40)],
            'mlp__activation': ['tanh', 'relu'],
            'mlp__max_iter': [200],
            'mlp__momentum': [0.9, 0.93, 0.96],
            'mlp__solver': ['sgd', 'adam'],
            'mlp__alpha': [0.0001],
            'mlp__learning_rate': ['constant','adaptive']
        }
    },
}

# initializing pipelines with its corrosponding steps
pipe = Pipeline(steps=Pipelines[model]['steps'])

# Parameters of pipelines can be set using '__' separated parameter names:
search = GridSearchCV(pipe, Pipelines[model]['params'],cv=10,verbose=2) 
search.fit(x_train, y_train) # fit the pipelins and search the best parameters if any
# outputs
print("Best parameter (CV score=%0.3f):" % search.best_score_) 
print(search.best_params_)
print(accuracy_score(y_pred=search.predict(x_test),y_true=y_test))


#save the model
filename = f'{Pipelines[model]["name"]}_rf_3_flexions.sav'
pickle.dump(search, open(filename, 'wb'))
# saves the test data set for further model performance analysis
np.save("./x_test.np",x_test)
np.save("./y_test.np",y_test)
