import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Run LSTM.')
parser.add_argument('path', help='File path')
parser.add_argument('--outputs', default=169, help="Output vector size", type=int)
parser.add_argument('--timesteps', default=1, help="Timesteps", type=int)
parser.add_argument('--train_perc', default=0.8, help="Percentage of the data used to train the model. Value between 0 and 1. Default:0.8", type=float)
parser.add_argument('--batch_size', default=1200, help="Batch size used to train LSTM",type=int)
parser.add_argument('--epochs', default=10, help="Input Values", type=int)
parser.add_argument('--preprocess', default="minmax_1", help="minmax or standarization")
parser.add_argument('--stateful', default=False, help="Stateful Keras LSTM", type=str2bool)
parser.add_argument('--verbose', default=False, help="Verbose Keras LSTM", type=str2bool)
args = parser.parse_args()

#Imports
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as metrics

import sys
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K
from PyEMD import EMD

dataPath = args.path
output_dim = args.outputs
timesteps = args.timesteps
train_perc = args.train_perc
batch_size = args.batch_size
epochs = args.epochs
preprocess = args.preprocess
stateful = args.stateful
verbose = args.verbose

filename = os.path.basename(dataPath).replace(".csv","")
file_path = "results/params_lstm_one/{}_scores_lstm_{}_outs_{}_timesteps_{}_batch_{}_preprocess_{}_stateful.csv".format(filename, output_dim,timesteps,batch_size, preprocess, stateful)

if os.path.exists(file_path):
	print("File Exists, Skipping")
	exit(0)


def create_data_cube(data,input_dim = 22, output_dim = 12, timesteps=720):
    m,input_dim = data.shape

    X = np.empty((m,timesteps,input_dim))
    y = np.empty((m,output_dim,input_dim))

    try:
        for i,t in enumerate(range(timesteps,m)):
            X[i,:,:] = data[t-timesteps:t, :]
            y[i,:] = data[t:t+output_dim,:]
    except:
        X = X[:i,:,:]
        y = y[:i,:,:]
    return X,y

def create_lstm(input_dim,output_dim,timesteps, nodes,loss='mean_squared_error',optimizer='adam',activation="tanh",recurrent_activation='hard_sigmoid', batch_size = 168,stateful=False):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(None,input_dim),activation=activation, recurrent_activation=recurrent_activation,stateful=stateful, batch_size=batch_size))
    model.add(Dense(input_dim))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def n_predict(model,X,input_dim = 22,steps=12, batch_size=168):
    y = np.empty((len(X),steps,input_dim))
    X_tmp = np.empty((X.shape[0], X.shape[1]+steps,X.shape[2]))
    X_tmp[:,:X.shape[1], :] = X
    for i in range(steps):
        X_iter = X_tmp[:,:X.shape[1]+i,:]
        y[:,i,:] = model.predict(X_iter,batch_size = batch_size)
        X_tmp[:,X.shape[1]+i,:] = y[:,i,:]
    return y


print("Reading Data")
#Data split parameters
data = np.loadtxt(dataPath)
input_dim = data.shape[1]

print("Preparing data")
X,y = create_data_cube(data,input_dim=input_dim, timesteps=timesteps, output_dim = output_dim)

trainlen1 = int(train_perc*len(X))
trainlen = int(train_perc*trainlen1)
print("Removed from train ", (trainlen%batch_size), " values")
trainlen = trainlen - (trainlen%batch_size)
vallen = trainlen1 - trainlen
print("Removed from validation ", (vallen%batch_size), " values")
vallen = vallen - (vallen%batch_size)

X_train,X_test = X[:trainlen], X[trainlen:trainlen+vallen]
y_train,y_test = y[:trainlen,0], y[trainlen:trainlen+vallen]

print("Preprocessing Data")

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
    minmax_out.fit(y_train)

    preproc_in = minmax_in
    preproc_out = minmax_out

else:
    print("Using Standarization")
    standarization_in = preprocessing.StandardScaler()
    standarization_out = preprocessing.StandardScaler()

    standarization_in.fit(X_train[:,0,:])
    standarization_out.fit(y_train)

    preproc_in = standarization_in
    preproc_out = standarization_out

for i in range(timesteps):
    X_train[:,i,:] = preproc_in.transform(X_train[:,i,:]) if preproc_in else X_train
    X_test[:,i,:] = preproc_in.transform(X_test[:,i,:]) if preproc_in else X_test
y_train = preproc_out.transform(y_train) if preproc_out else y_train


print("Creating Param List")
lstm_nodes = [32]
loss = ["mean_squared_error"]
activation = ["tanh", "sigmoid"]
recurrent_activation = ["sigmoid"]
param_grid = {"nodes":lstm_nodes,"loss":loss, "input_dim":[input_dim], "output_dim": [output_dim], "timesteps":[timesteps],
              "activation":activation, "recurrent_activation":recurrent_activation, "batch_size":[batch_size]}
params = ms.ParameterGrid(param_grid)

print("Evaluating Models")

scores = []
for param in params:
    print(param)
    np.random.seed(42)
    model = create_lstm(**param)
    model.fit(X_train, y_train,shuffle=False,verbose=verbose, epochs=epochs, batch_size=batch_size)

    y_approx = n_predict(model,X_test,input_dim = input_dim,batch_size=batch_size,steps=output_dim)
    for i in range(output_dim):
        y_approx[:,i,:] =  preproc_out.inverse_transform(y_approx[:,i,:])
    # score = metrics.mean_squared_error(y_test,y_approx)
    score = metrics.mean_squared_error( np.sum(y_test,axis=2), np.sum(y_approx,axis=2))
    print("Score Validation: ",score)
    param["score"] = score
    scores.append(param)

    del model
    K.clear_session()


scores = pd.DataFrame(scores)


scores["preproc"] = preprocess
os.makedirs("results/params_lstm_one",exist_ok=True)
scores.to_csv( "results/params_lstm_one/{}_scores_lstm_{}_outs_{}_timesteps_{}_batch_{}_preprocess_{}_stateful.csv".format(filename, output_dim,timesteps,batch_size, preprocess, stateful))
