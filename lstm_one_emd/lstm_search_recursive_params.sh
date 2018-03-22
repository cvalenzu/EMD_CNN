#!/bin/bash
export CUDA_VISIBLE_DEVICES=1


preproc=( std minmax1 minmax2 )
timesteps=( 24 48 72 168 336 672 )
for timestep in ${timesteps[@]};do
		for pre in ${preproc[@]};do
			python lstm_selecting_recursive_params.py  --outputs 169 --timesteps $timestep --epochs 10 --preprocess $pre --verbose t canela_emd.csv
		done
done
