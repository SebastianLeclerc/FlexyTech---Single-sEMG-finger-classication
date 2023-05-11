import pandas as pd
import numpy as np
from src.data.data_combination import combine_data
from src.data.data_processing import ADC_to_v, format_timestamps, normalize_data
import librosa
import os, shutil
import matplotlib.pyplot as plt
import scipy
from scipy.stats import kurtosis, skew
from librosa import display
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
from scipy.integrate import cumtrapz
from src.data.features_extraction import *

raw_data, labels, time_stamps = combine_data('./../data/al_data')
labels -= 1

voltage_values = []
time = []
for i in raw_data:
    voltage_values.append(ADC_to_v(i))
for i in time_stamps:
    time.append(format_timestamps(i))

n_sample = 7

win_size = 200
# plt.subplot(3,1,1)
f1 = np.lib.stride_tricks.sliding_window_view(voltage_values[n_sample], win_size).mean(axis=1)
# f2 = skewness(np.lib.stride_tricks.sliding_window_view(voltage_values[n_sample],win_size),axis=1)/f3
f3 = np.lib.stride_tricks.sliding_window_view(voltage_values[n_sample], win_size).max(axis=1)
f4 = kurtosis(np.lib.stride_tricks.sliding_window_view(voltage_values[n_sample], win_size), axis=1)
f2 = skew(np.lib.stride_tricks.sliding_window_view(voltage_values[n_sample], win_size), axis=1)
f3 = np.apply_along_axis(lambda x: x * (x > 0), 0, f4 * f2 > f3)
plt.plot(voltage_values[n_sample])
# plt.plot(f1)
# plt.plot(f2)
plt.plot(f3)
# plt.plot(f4)
plt.grid()

# plt.subplot(3,1,2)
# plt.plot(voltage_values[n_sample+1])
# plt.plot(np.convolve(voltage_values[n_sample+1],np.ones(win_size)/win_size,'valid'))
# plt.plot(np.lib.stride_tricks.sliding_window_view(abs(voltage_values[n_sample+1]),win_size).mean(axis=1))
# plt.plot(np.lib.stride_tricks.sliding_window_view(voltage_values[n_sample+1],win_size).max(axis=1))
# plt.plot(np.lib.stride_tricks.sliding_window_view(voltage_values[n_sample+1],win_size).min(axis=1))
# plt.grid()
# plt.subplot(3,1,3)
# plt.plot(voltage_values[n_sample+2])
# plt.plot(np.convolve(voltage_values[n_sample+2],np.ones(win_size)/win_size,'valid'))
# plt.plot(np.lib.stride_tricks.sliding_window_view(abs(voltage_values[n_sample+2]),win_size).mean(axis=1))
# plt.plot(np.lib.stride_tricks.sliding_window_view(voltage_values[n_sample+2],win_size).max(axis=1))
# plt.plot(np.lib.stride_tricks.sliding_window_view(voltage_values[n_sample+2],win_size).min(axis=1))
# plt.grid()

np.lib.stride_tricks.sliding_window_view(voltage_values[n_sample], 250, ).var(axis=1)


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filter(voltage_values, low_cut=1, high_cut=55, fs=1e3, ):
    '''
    band pass filter based on the scipy butter bandpass, allowing frequency between low cut and high cut to pass through.
    input : np.array of the data voltage values.
    low_cut : lowest frequency defaults at 1
    high cut : highest frequency in the band filter defaults at 55.
    '''
    return butter_bandpass_filter(voltage_values, lowcut=low_cut, highcut=high_cut, fs=fs)


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


fft_vals = abs(fft(voltage_values[n_sample], n=16))
fft_vals

plt.plot(librosa.fft_frequencies(sr=1000, n_fft=16), fft_vals[:len(fft_vals) // 2 + 1])

filtered = butter_bandpass_filter(voltage_values[n_sample], 1, 40, 1000)

plt.plot(filtered)
plt.grid()

plt.plot((filtered))
plt.plot(1 / np.pi * cumtrapz(filtered))

filtered_values = []
for i in voltage_values:
    filtered_values.append(filter(i, high_cut=60))

plt.subplot(1, 2, 1)
plt.title(f"{labels[n_sample]} filtered")
plt.plot(filtered)
plt.grid()
plt.subplot(1, 2, 2)
plt.title(f"{labels[n_sample]} raw")
plt.plot(voltage_values[n_sample])
plt.grid()

features = [root_mean_squared, iemg, mean_absolute_value, variance, wave_form_length, mean, median, skewness, kurt,
            zero_crossing_rate]
df = extract_features(voltage_values, label=labels, features_no=10, overlapping_percentage=0.25,
                      features_funcs=features)
df

df.columns

import seaborn as sns
from sklearn.decomposition import PCA
import pickle as pk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, normalize, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import Sequential

X, y = df.drop('label', axis=1), df['label']
X_scaled = minmax_scale(X)

pca = PCA(n_components=25)
pca.fit(X_scaled)
components = pca.transform(X_scaled)

x_train, x_test, y_train, y_test = train_test_split(components, y, train_size=0.7)
x_train.shape

with open('./pca.pkl', 'wb') as pkl:
    pk.dump(pca, pkl)

# parameter 1 
encoding_dim = 100
# ncol # without the label column
ncol = x_train.shape[1:][0]
ncol

model = Sequential()
model.add(keras.layers.Input(shape=x_train.shape[1:]))
model.add(keras.layers.Dense(15))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(5))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(3, activation='softmax'))

loss_fn = loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss_fn, metrics=['accuracy'], )

model.summary()

model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=200, verbose=False)

from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train, y_train)

predictions_ann = np.argmax(model.predict(x_test), axis=1)
predictions_svc = svm.predict(x_test)
from sklearn.metrics import classification_report

print(classification_report(y_true=y_test, y_pred=predictions_ann))

model.save('ann.h5')

import inspect

print(inspect.getsource(ADC_to_v))

dataMax = np.max([i for i in map(len, raw_data)])
out_size = 128
max_features = 20000

X = np.array(raw_data)
y = np.array(labels)

for i, element in enumerate(X):
    temp = np.zeros(dataMax)
    temp[:len(element)] += element[0]
    X[i] = np.array(temp, dtype='float32')

X = tf.convert_to_tensor([tf.convert_to_tensor(i) for i in X])

X = np.expand_dims(X, axis=2)

model = Sequential()
model.add(keras.layers.Input(shape=X.shape[1:]))
model.add(keras.layers.Masking(mask_value=0))
model.add(keras.layers.Conv1D(2, ))
model.add(keras.layers.Dense(3, activation='softmax'))
loss_fn = loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss_fn, metrics=['accuracy'], )

model.build()

model.summary()

x_train, x_test, y_train, y_test = train_test_split(X, y)

model.fit(x_train, y_train)
