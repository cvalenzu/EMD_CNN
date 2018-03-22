import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Run LSTM.')
parser.add_argument('params_path', help='Best params file path')
parser.add_argument('output', help='Output path')
parser.add_argument('--epochs', default=100, help="Input Values", type=int)
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

from PyEMD import EMD
import progressbar
import numpy.polynomial.polynomial as poly


def create_data_cube(data,input_dim=24, output_dim = 12, timesteps=720):
    m = len(data)

    A = np.empty((m,input_dim))
    B = np.empty((m, output_dim))

    try:
        for i in range(m):
            window = data[i+input_dim:i:-1]
            A[i,:] = window
            B[i,:] = data[i+input_dim:i+input_dim+output_dim]
    except:
        A = A[:i,:]
    X = np.empty((i-timesteps, timesteps, input_dim))
    y = np.empty((i-timesteps, output_dim))
    for j in range(timesteps):
        X[:,j,:] = A[j:i-(timesteps-j),:]

    for a in range(i-(timesteps)):
        y[a,:] = B[a+timesteps,:]
    A = None
    B = None
    return X,y


output = args.output if args.output[-1] == "/" else args.output + "/"
train_perc = 0.8
batch_size = 1200
epochs = args.epochs
verbose = args.verbose


os.makedirs(args.output, exist_ok=True)
scores = []
params = pd.read_csv(args.params_path,index_col=0)
def create_lstm(input_dim,output_dim,timesteps, nodes,loss='mean_squared_error',optimizer='adam',activation="tanh",recurrent_activation='hard_sigmoid', batch_size = 168,stateful=False):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(None,input_dim),activation=activation, recurrent_activation=recurrent_activation,stateful=stateful, batch_size=batch_size))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def n_predict(model,X,steps=12, batch_size=168):
    y = np.empty((len(X),steps))
    X_tmp = np.empty((X.shape[0], X.shape[1]+steps,X.shape[2]))
    X_tmp[:,:X.shape[1], :] = X
    for i in range(steps):
        X_iter = X_tmp[:,:X.shape[1]+i,:]
        y[:,i] = model.predict(X_iter,batch_size = batch_size)[:,0]
        X_tmp[:,X.shape[1]+i,0] = y[:,i]
    return y


#Load data
data = pd.Series.from_csv("../data/canela.csv")
for index,param in params.iterrows():
    preprocess = param["preproc"]
    input_dim = param["input_dim"]
    timesteps = param["timesteps"]
    output_dim = param["output_dim"]
    X,y = create_data_cube(data, input_dim=input_dim,timesteps=timesteps, output_dim = output_dim)

    n = X.shape[0]
    timesteps = X.shape[1]
    input_dim = X.shape[2]

    trainlen = int(train_perc*len(X))
    print("Removed from train ", (trainlen%batch_size), " values")
    trainlen = trainlen - (trainlen%batch_size)
    testlen = n - (n%batch_size)

    X_train,X_test = X[:trainlen], X[trainlen:trainlen+testlen]
    y_train,y_test = y[:trainlen,0].reshape(-1,1), y[trainlen:trainlen+testlen,:]

    m = X_test.shape[0]
    m = m - (m%batch_size)

    X_test = X_test[:m,:]
    y_test = y_test[:m,:]

    print(X_test.shape[0]%batch_size) 
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
    y_train= preproc_out.transform(y_train) if preproc_out else y_train


    param = param.drop(["score","preproc"])
    np.random.seed(42)
    model = create_lstm(**param)

    print("Fitting Model")
    model.fit(X_train, y_train,shuffle=False,verbose=verbose, epochs=epochs, batch_size=batch_size)
    y_approx = n_predict(model,X_test,batch_size=batch_size,steps=output_dim)
    
    for j in range(output_dim):
        y_approx[:,j] = preproc_out.inverse_transform(y_approx[:,j].reshape(-1,1)).reshape(-1)
    
    score = metrics.mean_squared_error(y_test,y_approx)

    print("Score Validation: ",score)
    param["score"] = score

    imfs = param["input_dim"]-1
    timesteps = param["timesteps"]

    y_approx = pd.DataFrame(y_approx)
    y_approx.to_csv(output+"y_approx_{}_imfs_{}_timesteps.csv".format(imfs,timesteps))

    y_test = pd.DataFrame(y_test)
    y_test.to_csv(output+"y_test_{}_imfs_{}_timesteps.csv".format(imfs,timesteps))

    model.save(output+"model_{}_imfs_{}_timesteps.csv".format(imfs,timesteps))

    del model
    K.clear_session()
