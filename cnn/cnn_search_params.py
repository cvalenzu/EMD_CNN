import argparse

parser = argparse.ArgumentParser(description='Run LSTM.')
parser.add_argument('path', help='File path')
parser.add_argument('output_path', help='File path')
args = parser.parse_args()

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from keras import backend as K

import sklearn.model_selection as ms
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing

import pandas as pd
import os




batch_size = 1200
epochs = 10
verbose = True
train_perc = 0.8

output_path = args.output_path
output_path = output_path if output_path[-1] == "/" else output_path+"/"
path = args.path
filename = os.path.basename(path).split("_")
source = filename[1]

preprocess= "minmax"

os.makedirs(output_path,exist_ok=True)


X = np.load(path)
y =  np.load(path.replace("X","y"))


n = X.shape[0]
timesteps = X.shape[1]
imfs = X.shape[2]

train_len = int(train_perc*n)
val_len = int(train_perc*train_len)
X_train, X_test = X[:val_len], X[val_len:train_len]
y_train, y_test = y[:val_len], y[val_len:train_len]


if "minmax" in preprocess:
    print("Using Minmax Scaler with feature range ", end="")
    if "1" in preprocess:
        print(" (0,1) ")
        feature_range=(0,1)
    else:
        print( " (-1,1) " )
        feature_range=(-1,1)
    minmax_in = preprocessing.MinMaxScaler(feature_range=feature_range)
    minmax_out = preprocessing.MinMaxScaler(feature_range=feature_range)

    minmax_in.fit(X_train[:,0,:])
    minmax_out.fit(y_train.reshape(-1,1))

    preproc_in = minmax_in
    preproc_out = minmax_out

else:
    print("Using Standarization")
    standarization_in = preprocessing.StandardScaler()
    standarization_out = preprocessing.StandardScaler()

    standarization_in.fit(X_train[:,0,:])
    standarization_out.fit(y_train[:,0].reshape(-1,1))

    preproc_in = standarization_in
    preproc_out = standarization_out

for i in range(timesteps):
    X_train[:,i,:] = preproc_in.transform(X_train[:,i,:]) if preproc_in else X_train
    X_test[:,i,:] = preproc_in.transform(X_test[:,i,:]) if preproc_in else X_test
y_train_process = preproc_out.transform(y_train[:,0].reshape(-1,1)) if preproc_out else y_train[:,0].reshape(-1,1)



print("Creating Param List")
activations = ["relu", "tanh", "sigmoid"]
maxpool = [True,False]
filters = np.arange(2,10)
lags = np.arange(6,49,6)
param_grid = {"timesteps":[timesteps], "imfs": [imfs],"activation": activations,
              "maxpool":maxpool,"filters":filters,"lags":lags}
params = ms.ParameterGrid(param_grid)


def create_model(filters, lags,timesteps,imfs, activation = "relu", maxpool=False):
    lags = int(lags)
    filters = int(filters)
    timesteps = int(timesteps)
    model = Sequential()
    model.add(Conv1D(filters=filters,kernel_size=lags, input_shape=(timesteps,imfs),activation=activation))
    if maxpool:
        model.add(MaxPooling1D(lags))
    model.add(Flatten())
    model.add(Dense(1))
    return model



print("Evaluating Models")
y_metrics = np.zeros(len(params))
for i,param in enumerate(params):
    print(param)
    y_approx = np.zeros_like(y_test)
    for i in range(y_test.shape[1]):
        np.random.seed(42)
        model = create_model(**param)
        model.compile(loss='mean_squared_error',
                  optimizer='adam')

        model.fit(X_train,y_train[:,i], batch_size=batch_size, epochs=epochs,verbose=verbose)
        models.append(model)

        y_iter = preproc_out.inverse_transform(model.predict(X_test))
        y_approx[:,i] = y_iter[:,0]
        del model
        K.clear_session()
    y_metric = metrics.mean_squared_error(y_test,y_approx)
    y_metrics[i] = y_metric


p = []
for param in params:
    p.append(param)
p = pd.DataFrame(p)


p["score"] = y_metrics

p.to_csv(output_path+"{}_{}_imfs_{}_timesteps.csv".format(source,imfs,timesteps))
