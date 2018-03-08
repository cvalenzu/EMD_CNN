#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


files=`ls -1 /mnt/cvalenzu/EMDS/canela/X*`
preproc=( std minmax1 minmax2 )
for file in ${files[@]};do
	for pre in ${preproc[@]};do
		python cnn_search_params.py $file results --preproc $pre
	done
done
