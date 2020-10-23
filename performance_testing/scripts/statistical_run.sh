#!/bin/bash

if [ -z "$1" ]
  then
    echo "Input the path to executable file"
  exit 1
fi

#The input parameter is the location of the executable
echo "Start statistical runs for $1"

pT=1.0
nTracks=(10) #(100 1000 5000 10000 50000 100000) 
gridSizes=('40')
blockSizes=('8*8')

# add variable B field 
#-b "../../acts-data/MagneticField/ATLAS/ATLASBField_xyz.txt"

for i in ${nTracks[@]}; do
	for ((j=0; j<${#gridSizes[@]};++j)); do
		for cnt in {1..2}; do
                	echo "Run $cnt for ${i} tracks with gridSize=${gridSizes[j]} and blockSize=${blockSizes[j]}"
                	./$1 -t ${i} -p ${pT} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]}
                	sleep 1;
		done
        done;
done;


