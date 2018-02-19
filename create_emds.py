import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Process timeseries with EMD.')
parser.add_argument('--imfs', '--im', type=int, default=1,help='max imfs from EMD')
parser.add_argument('--timesteps', '--t', type=int, default=24, help='timesteps for secuense (default=24)')
parser.add_argument('--outputs', '--o', type=int, default=169, help='values predited for secuense (default=168)')
parser.add_argument('input_path', help="input path for the timeseries.")
parser.add_argument('output_path', help="output directory.")
parser.add_argument("--debug",type=str2bool, default=False)
args = parser.parse_args()

from PyEMD import EMD
import pandas as pd
import numpy as np
import progressbar
import os

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

print("Creating EMD for {} with {} imfs and {} timesteps".format(os.path.basename(args.input_path), args.imfs,args.timesteps))
timesteps=args.timesteps
emd_dim = args.imfs
data = pd.Series.from_csv(args.input_path).values
if args.debug:
    data = data[:1000]
print("splitting data...",end="")
X,y = create_data_cube(data, input_dim=1, output_dim=args.outputs, timesteps=timesteps)
print("end")
X_new = np.empty((X.shape[0], X.shape[1],emd_dim+2))
emd = EMD()
bar = progressbar.ProgressBar(max_value = X.shape[0])
print("decomposing data...")
for i in range(X.shape[0]):
    timeseries = X[i,:,0]
    X_new[i,:,0] = X[i,:,0]
    try:
        emds = emd.emd(timeseries,max_imf = emd_dim).T
        X_new[i,:,1:] = emds
        bar.update(i)
    except ValueError:
        f = open("not_processed.csv", "a+")
        filename = os.path.basename(args.input_path).replace(".csv","")
        f.write("{},{},{}\n".format(filename,emd_dim,timesteps))
        f.close()


filename = os.path.basename(args.input_path).replace(".csv","")
base_dir = args.output_path if args.output_path[-1] == "/" else args.output_path + "/"

print("saving results..",end="")
os.makedirs(base_dir, exist_ok=True)

np.save(base_dir+"X_{}_{}_imfs_{}_timesteps".format(filename,emd_dim+1,timesteps),X_new)
np.save(base_dir+"y_{}_{}_imfs_{}_timesteps".format(filename,emd_dim+1,timesteps),y)
print("finished")
