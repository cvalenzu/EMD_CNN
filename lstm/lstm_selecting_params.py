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
parser.add_argument('output', help='Output path')
parser.add_argument('--epochs', default=10, help="Input Values", type=int)
parser.add_argument('--preprocess', default="minmax_1", help="minmax or standarization")
parser.add_argument('--verbose', default=False, help="Verbose Keras LSTM", type=str2bool)
args = parser.parse_args()

#Imports
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K


dataPath = args.path
output = args.output if args.output[-1] == "/" else args.output + "/"
train_perc = 0.8
batch_size = 1200
epochs = 10
preprocess = args.preprocess
stateful = False
verbose = args.verbose


os.makedirs(args.output, exist_ok=True)

filename = os.path.basename(dataPath).replace(".csv","").split("_")
source = filename[1]


print("Reading Data")
X = np.load(args.path)
y = np.load(args.path.replace("X", "y"))

n = X.shape[0]
timesteps = X.shape[1]
input_dim = X.shape[2]

trainlen1 = int(train_perc*len(X))
trainlen = int(train_perc*trainlen1)
print("Removed from train ", (trainlen%batch_size), " values")
trainlen = trainlen - (trainlen%batch_size)
vallen = trainlen1 - trainlen
print("Removed from validation ", (vallen%batch_size), " values")
vallen = vallen - (vallen%batch_size)

X_train,X_test = X[:trainlen], X[trainlen:trainlen+vallen]
y_train,y_test = y[:trainlen,0].reshape(-1,1), y[trainlen:trainlen+vallen,0].reshape(-1,1)

def create_lstm(input_dim,output_dim,timesteps, nodes,loss='mean_squared_error',optimizer='adam',activation="tanh",recurrent_activation='hard_sigmoid', batch_size = 168,stateful=False):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(timesteps,input_dim),activation=activation, recurrent_activation=recurrent_activation,stateful=stateful, batch_size=batch_size,unroll=True))
    model.add(Dense(output_dim))
    model.compile(loss=loss, optimizer=optimizer)
    return model

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
lsm_nodes = np.arange(10,33,2)
loss = ["mean_squared_error"]
activation = ["tanh", "sigmoid"]
recurrent_activation = ["sigmoid"]
param_grid = {"nodes":lsm_nodes,"loss":loss, "input_dim":[input_dim], "output_dim": [1], "timesteps":[timesteps],
              "activation":activation, "recurrent_activation":recurrent_activation, "batch_size":[batch_size]}
params = ms.ParameterGrid(param_grid)


print("Evaluating Models")

scores = []
for param in params:
    print(param)
    np.random.seed(42)
    model = create_lstm(**param)

    model.fit(X_train, y_train,shuffle=False,verbose=verbose, epochs=epochs, batch_size=batch_size)
    try:
        y_approx_train = preproc_out.inverse_transform(model.predict(X_train,batch_size=batch_size))

        y_approx = preproc_out.inverse_transform(model.predict(X_test,batch_size=batch_size))
        score = metrics.mean_squared_error(y_test[:,0],y_approx)

        print("Score Validation: ",score)
        param["score"] = score
        scores.append(param)



    except Exception as e:
        print("Error")
        print(str(e))
        continue

    del model
    K.clear_session()

scores = pd.DataFrame(scores)



scores["preproc"] = preprocess
scores.to_csv(output+"{}_{}_imfs_{}_timesteps.csv".format(source, input_dim, timesteps))
