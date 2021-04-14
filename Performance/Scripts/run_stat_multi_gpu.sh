#!/bin/bash

if [ -z "$1" ]
  then
    echo 1>&2 "Usage: $0 <EXECUTABLE> <MACHINE>, e.g. $0 <build>/Run/GPU/KalmanFitterGPUTest"
  exit 1
fi

#The input parameter is the location of the executable
echo "Start statistical runs for $1"

nStreams=(1) 

nTracks=(5 10 50 100 500 1000 5000 10000 50000 100000) 
gridSizes=('5120')  #'100000')
blockSizes=( '8x8')
sharedMemory=0

for i in ${nTracks[@]}; do
	for ((j=0; j<${#gridSizes[@]};++j)); do
            for k in ${nStreams[@]}; do
		for cnt in {1..10}; do
			echo "Run $cnt for ${i} tracks with gridSize=${gridSizes[j]} and blockSize=${blockSizes[j]} and ${k} streams:"
                        
			echo " multi gpu  $1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]} -s ${sharedMemory} -u 1"
			$1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]} -s ${sharedMemory} -u 1
	                
                        echo " single gpu  $1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]} -s ${sharedMemory} -u 0" 
			$1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]} -s ${sharedMemory} -u 0
        
			sleep 1;
		done
	    done
        done;
done;
