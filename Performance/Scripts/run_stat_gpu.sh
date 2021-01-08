#!/bin/bash

if [ -z "$1" ]
  then
    echo 1>&2 "Usage: $0 <EXECUTABLE> <MACHINE>, e.g. $0 <build>/Run/GPU/KalmanFitterGPUTest"
  exit 1
fi

#The input parameter is the location of the executable
echo "Start statistical runs for $1"

nTracks=(5 10 50 100 500 1000 5000 10000) 
nStreams=(1 4) 
gridSizes=('20000')
blockSizes=('8*8')

for i in ${nTracks[@]}; do
	for ((j=0; j<${#gridSizes[@]};++j)); do
            for k in ${nStreams[@]}; do
		for cnt in {1..5}; do
			echo "Run $cnt for ${i} tracks with gridSize=${gridSizes[j]} and blockSize=${blockSizes[j]} and ${k} streams:"
                        
			echo " 1 track per thread: $1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]}"
                        # 1 track per thread	
			$1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]}
	                
			# 1 block per thread, always 8** threads per block	
                        echo " 1 block per thread: $1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -s 1" 
			$1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -s 1
                
			sleep 1;
		done
	    done
        done;
done;
