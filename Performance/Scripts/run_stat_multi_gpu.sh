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
gridSizes=( '100000')
blockSizes=( '8x8')

### griSizes and blockSizes list for other blockSizes configurations ###
#nTracks=(100000) 
#1) lauch bounds (256, 2)
#gridSizes=('100000x1x1' '100000x1x1' '100000x1x1' '5120x1x1' '5120x1x1' '5120x1x1')
#blockSizes=('16x16x1' '64x1x1' '256x1x1' '16x16x1' '64x1x1' '256x1x1')
#2) lauch bounds (1024, 2)
#gridSizes=('100000x1x1' '100000x1x1' '5120x1x1' '5120x1x1')
#blockSizes=('1024x1x1' '32x32x1' '1024x1x1' '32x32x1')
############################################################

for i in ${nTracks[@]}; do
	for ((j=0; j<${#gridSizes[@]};++j)); do
            for k in ${nStreams[@]}; do
		for cnt in {1..5}; do
			echo "Run $cnt for ${i} tracks with gridSize=${gridSizes[j]} and blockSize=${blockSizes[j]} and ${k} streams:"
                        
			echo " multi gpu  $1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]} -s 1 -u"
			$1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]} -s 1 -u 1
	                
                      #  echo " single gpu  $1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -s 1" 
			$1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]} -s 1 
                
			sleep 1;
		done
	    done
        done;
done;
