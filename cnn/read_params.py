import argparse

parser = argparse.ArgumentParser(description='Run LSTM.')
parser.add_argument('path', help='File path')
args = parser.parse_args()

import glob
import pandas as pd

files = glob.glob(args.path+"/*")

results = []
for f in files:
	df = pd.read_csv(f,index_col = 0)
	df = df.sort_values("score")	
	best = df.iloc[0,:]
	results.append(best)
results = pd.DataFrame(results)
