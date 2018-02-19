#!/bin/bash

basePath=data/
paths=( canela.csv  monte_redondo.csv  totoral.csv )
out_path=EMDS

output=168
timesteps=( 24 48 72 168 336 672 1344 )
max_imfs=( 0 1 2 4 6 )
debug=f
nproc=`nproc`
rm args.dat 2>/dev/null
for timestep in ${timesteps[@]};do
	for path in ${paths[@]}; do
		for imf in ${max_imfs[@]};do
			echo "create_emds.py $basePath$path $out_path --debug $debug --timesteps $timestep --imfs $imf" >> args.dat
		done
	done
done

<args.dat xargs -L1 -P$nproc python
echo "Finish"
