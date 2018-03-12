#!/bin/bash
export CUDA_VISIBLE_DEVICES=1


preproc=( std minmax1 minmax2 )
timesteps=( 24 48 72 168 336 672 )
imfs=( 1 2 3 4 6 7 )
for timestep in ${timesteps[@]};do
	for imf in ${imfs[@]};do
		for pre in ${preproc[@]};do
			file=/mnt/cvalenzu/EMDS/canela/X_canela_${imf}_imfs_${timestep}_timesteps.npy
			echo $file $pre
			python lstm_selecting_params.py $file results --preproc $pre --verbose t
		done
	done
done
