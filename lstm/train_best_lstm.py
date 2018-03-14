import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Run LSTM.')
parser.add_argument('emd_path', help='EMD folder path')
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


dataPath = args.emd_path if args.emd_path[-1] == "/" else args.emd_path + "/"
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
    model.add(Dense(output_dim))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def n_predict(model,X,h=169):
    n = X.shape[0]
    imfs = X.shape[2] - 2
    timesteps = X.shape[1]
    y_approx = np.empty((n, h))
    X_new = np.empty((n,timesteps+h,imfs+2))
    X_new[:,:timesteps,:] = X
    emd = EMD()
    bar = progressbar.ProgressBar(max_value = h*n)
    bar.start()
    iter = 0
    for i in range(h):
        predict = model.predict(X_new[:,:timesteps+i,:],batch_size=batch_size)
        X_new[:,timesteps+i,0] = predict[:,0]
        for j in range(n):
            value = X_new[j,timesteps+i,0]
            for imf in range(1,X.shape[2]):
                a = X_new[j,:timesteps+i,0]
                b = X_new[j,:timesteps+i,imf]
                coefs = poly.polyfit(a, b, 3)
                ffit = poly.polyval(value, coefs)
                X_new[j,timesteps+i,imf] = ffit
            iter +=1
            bar.update(iter)

        y_approx_i = preproc_out.inverse_transform(predict)
        y_approx[:,i] = y_approx_i[:,0]
    return y_approx

for index,param in params.iterrows():
    # if param["timesteps"] <= 24 or param["input_dim"]<=2:
    #     continue
    print(param)
    preprocess = param["preproc"]
    print("Reading Data")
    dataFile = dataPath+"X_canela_{}_imfs_{}_timesteps.npy".format(param["input_dim"]-1, param["timesteps"])
    print(dataFile)
    try:
        X = np.load(dataFile)
        y = np.load(dataFile.replace("X", "y"))
    except:
        print("File:{} Not found, skipping".format(os.path.basename(dataPath)))
        continue

    n = X.shape[0]
    timesteps = X.shape[1]
    input_dim = X.shape[2]

    trainlen = int(train_perc*len(X))
    print("Removed from train ", (trainlen%batch_size), " values")
    trainlen = trainlen - (trainlen%batch_size)
    testlen = n - (n%batch_size)

    X_train,X_test = X[:trainlen], X[trainlen:testlen]
    y_train,y_test = y[:trainlen,0].reshape(-1,1), y[trainlen:testlen,:]

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


    param = param.drop(["score","preproc"])
    np.random.seed(42)
    model = create_lstm(**param)

    print("Fitting Model")
    model.fit(X_train, y_train,shuffle=False,verbose=verbose, epochs=epochs, batch_size=batch_size)
    y_approx = n_predict(model,X_test)

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
