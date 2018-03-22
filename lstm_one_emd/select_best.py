import glob
import pandas as pd


files = glob.glob("results/params_lstm_one/*.csv")

dfs = []

for file in files:
	df = pd.read_csv(file,index_col = 0)
	best = df.sort_values("score").iloc[0,:]
	dfs.append(best)
dfs = pd.DataFrame(dfs)

bests = []
for (imfs,timestep),df in dfs.groupby(["input_dim","timesteps"]):
	best = df.sort_values("score").iloc[0,:]
	bests.append(best)

bests = pd.DataFrame(bests)


bests.to_csv("best_params.csv")
