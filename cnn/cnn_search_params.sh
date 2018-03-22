#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


timesteps=( 24 48 72 168 336 672 )
preproc=( std minmax1 minmax2 )
for ts in ${timesteps[@]};do
	for pre in ${preproc[@]};do
		python cnn_params.py  ../data/canela.csv  results_cnn --timesteps $ts --train_perc 0.8 --batch_size 1200 --preproc $pre --epochs 10 --verbose t
		python cnn_emd_params.py  ../lstm_one_emd/canela_emd.csv  results_cnn_emd --timesteps $ts --train_perc 0.8 --batch_size 1200 --preproc $pre --epochs 10 --verbose t
	done
done
