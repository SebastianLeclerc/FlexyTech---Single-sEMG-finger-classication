from src import RepeatedKFold, GridSearchCV, KNeighborsClassifier, SVC
import tensorflow as tf
from tensorflow import keras
from keras import Sequential

def train_model_ann(X_train,y_train,model_architecture,):
    '''
    train ann model.
    input dimension
    classes labels to predict.
    model architecture list of number of nodes in hidden layers.
    example 
    if model is architecture [4,3] then the model has 2 hidden layers each have 4 and 3 nodes respectively.


    '''
    model = Sequential()
    model.add(keras.layers.Input(shape=X_train.shape[1:]))
    for i in model_architecture:
        model.add(keras.layers.Dense(i))
        model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(3,activation = 'softmax'))


    return model

