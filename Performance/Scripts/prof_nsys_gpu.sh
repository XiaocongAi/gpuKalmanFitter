!/bin/bash

if [ $# -ne 2 ]
  then
    echo 1>&2 "Usage: $0 <EXECUTABLE> <MACHINE>, e.g. $0 <build>/Run/GPU/KalmanFitterGPUTest Tesla_V100-SXM2-16GB"
  exit 1
fi

#The input parameter is the location of the executable
echo "Start profiling for $1 on $2"

nsysOption="nsys profile --stats=true --export=sqlite -t nvtx,cuda"

launch_bounds="None"
#launch_bounds="1024-1"

nStreams=(1 4) 

nTracks=(10000) 
gridSizes=('5120x1x1')
blockSizes=('8x8x1')


for i in ${nTracks[@]}; do
	for ((j=0; j<${#gridSizes[@]};++j)); do
            for k in ${nStreams[@]}; do
		#Only profiling once
		for cnt in {1..1}; do
			echo "Profiling $cnt for ${i} tracks with gridSize=${gridSizes[j]} and blockSize=${blockSizes[j]} and ${k} streams:"

                        # 1 track per thread
			nsysOutput1=Report_nsys_$2_nTracks_${i}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_0_launchBounds_${launch_bounds}	
			echo "prof 1 track per thread: ${nsysOption} -o ${nsysOutput1} $1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]}"
			${nsysOption} -o ${nsysOutput1} $1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -b ${blockSizes[j]}
	                
			# 1 block per thread, always 8x8 threads per block. Profiling for only block size 64	
		        if [[ ${blockSizes[j]} == "8x8x1" || ${blockSizes[j]} == "64x1x1" ]]; then	
			   nsysOutput2=Report_nsys_$2_nTracks_${i}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_1_launchBounds_${launch_bounds}	
                           echo "prof 1 block per thread: ${nsysOption} -o ${nsysOutput2} $1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -s 1" 
		    	   ${nsysOption} -o ${nsysOutput2} $1 -t ${i} -e ${k} -d "gpu" -o 0 -g ${gridSizes[j]} -s 1
                        fi

			sleep 1;
		done
	    done
        done;
done;
