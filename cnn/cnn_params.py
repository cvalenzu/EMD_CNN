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
parser.add_argument('outpath', help='File path')

parser.add_argument('--inputs', default=1, help="Input vector size", type=int)
parser.add_argument('--outputs', default=169, help="Output vector size", type=int)
parser.add_argument('--timesteps', default=24, help="Timesteps", type=int)
parser.add_argument('--train_perc', default=0.8, help="Percentage of the data used to train the model. Value between 0 and 1. Default:0.8", type=float)
parser.add_argument('--batch_size', default=1200, help="Batch size used to train LSTM",type=int)
parser.add_argument('--epochs', default=10, help="Input Values", type=int)
parser.add_argument('--preprocess', default="minmax_1", help="minmax or standarization")
parser.add_argument('--verbose', default=False, help="Verbose Keras", type=str2bool)
args = parser.parse_args()

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten,Reshape,Input,UpSampling1D
from keras.callbacks import TensorBoard

from keras import backend as K

import sklearn.model_selection as ms
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing

import pandas as pd
import os
try:
    import progressbar
except:
    progressbar = None


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


def n_predict(model, X,output_dim=169,batch_size=1200):
    if progressbar:
        bar = progressbar.ProgressBar(max_value=output_dim)
        bar.start()
    n,lags,input_dim = X.shape
    y_approx = np.empty((n,output_dim))
    for i in range(output_dim):
        y_i = model.predict(X,batch_size=batch_size)

        y_approx[:,i] = y_i[:,0]

        X = np.roll(X, 1, axis=1)
        X[:,-1,:] = y_i
        if progressbar:
            bar.update(i+1)
    return y_approx

def create_cnn(layers, filters, lags,timesteps,imfs, activation = "relu"):
    lags = int(lags)
    filters = int(filters)
    timesteps = int(timesteps)


    input_ts = Input(shape=(timesteps,imfs))
    x = Conv1D(filters=filters,kernel_size=lags, activation=activation, padding="same")(input_ts)
    x = MaxPooling1D(2, padding="same",strides=2)(x)
    for i in range(layers-2):
        if x.get_shape().as_list()[1]/2 % 2 != 0:
            break
        x = Conv1D(filters=filters,kernel_size=lags, activation=activation, padding="same")(x)
        x = MaxPooling1D(2, padding="same")(x)


    x = Conv1D(filters=1,kernel_size=lags, activation=activation, padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)

    flat = Flatten()(x)


    out = Dense(1,activation="linear")(flat)
    cnn = Model(input_ts,out)
    return cnn

batch_size = args.batch_size
epochs = args.epochs
verbose = args.verbose
train_perc = args.train_perc
timesteps = args.timesteps
output_dim = args.outputs
output_path = args.outpath
output_path = output_path if output_path[-1] == "/" else output_path+"/"
path = args.path
source = "canela"

preprocess= args.preprocess

os.makedirs(output_path,exist_ok=True)

data = pd.Series.from_csv(path)
X, y = create_data_cube(data,input_dim=1, output_dim=output_dim, timesteps = timesteps)


n = X.shape[0]
timesteps = X.shape[1]
imfs = X.shape[2]

train_len = int(train_perc*n)
val_len = int(train_perc*train_len)
X_train, X_test = X[:val_len], X[val_len:train_len]
y_train, y_test = y[:val_len,0].reshape(-1,1), y[val_len:train_len]

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


activations = ["tanh", "sigmoid"]#["relu", "tanh", "sigmoid"]
filters = np.arange(2,10)
lags = np.arange(6,49,6)
param_grid = {"timesteps":[timesteps], "imfs": [imfs],"activation": activations,
              "filters":filters,"lags":lags, "layers":[int(np.log2(timesteps))]}
params = ms.ParameterGrid(param_grid)

print("Training Models")
scores = []
for param in params:
    print(param)
    model = create_cnn(**param)
    model.compile(optimizer="adadelta", loss="mean_squared_error")


    model.fit(X_train,y_train,epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False)

    y_approx = n_predict(model,X_test,output_dim=output_dim)

    try:
        for i in range(output_dim):
            y_approx[:,i] = preproc_out.inverse_transform(y_approx[:,i].reshape(-1,1)).reshape(-1)
    except ValueError:
        print("Inverse Transform Error")
        K.clear_session()
        del model
        continue


    score = metrics.mean_squared_error(y_test,y_approx)


    param["score"] = score
    scores.append(param)
    print(score)
    del model
    K.clear_session()

print("Saving model")
scores = pd.DataFrame(scores)
scores["preproc"] = preprocess
scores.to_csv(output_path+"canela_scores_cnn_{}_timesteps_{}_preprocess.csv".format(timesteps, preprocess))
