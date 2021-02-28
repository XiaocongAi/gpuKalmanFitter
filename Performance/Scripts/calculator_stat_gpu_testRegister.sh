#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Missing boolean for shared memory or not"
  exit 1
fi

# check if the plotData and results directory exist
if [ ! -d "results" ]; then
  echo "Directory 'results' DOES NOT exists."
  exit 1
fi
if [ ! -d "plotData" ]; then
   echo "Directory 'plotData' DOES NOT exists. Creating the 'plotData' directory." 
   mkdir plotData   
fi

precision=float


machines=("Tesla_V100-SXM2-16GB")
nTracks=(10000)
nStreams=(1)

#1) This is for 8x8, 16x16 and 32x32
#launchBounds=('1024-2' '1024-1')
#numRegisterSuffix=('_maxRegistersPerThread_32' '_maxRegistersPerThread_64')
#gridSizes=('5120x1x1' '5120x1x1' '5120x1x1')
#blockSizes=('8x8x1' '16x16x1' '32x32x1')

#2) This is only for 8x8 and 16x16
launchBounds=('256-2')
numRegisterSuffix=('_maxRegistersPerThread_128')
gridSizes=('5120x1x1' '5120x1x1')
blockSizes=('8x8x1' '16x16x1')

################################

##### Tesla_P100-PCIE-16GB #####
### griSizes and blockSizes list for other blockSizes configurations ###
#nTracks=(10000)
#gridSizes=('100000x1x1' '5120x1x1')
#blockSizes=('8x8x1' '8x8x1')
###############################


# helper functions to get mean and sigma
getMean(){
awk 'BEGIN{s=0;}{s=s+$1;}END{print s/NR;}' $1
}

getSigma(){
awk '{delta = $1 - avg; avg += delta / NR; mean2 += delta * ($1 - avg); } END { print sqrt(mean2 / NR); }' $1
}


for ((m=0; m<${#machines[@]};++m)); do
    for ((j=0; j<${#gridSizes[@]};++j)); do
        for k in ${nStreams[@]}; do
          for ((d=0; d<${#launchBounds[@]};++d)); do
	    if [ $1 -eq 1 ]; then 
	      output=./plotData/${precision}/Results_timing_${machines[m]}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_8x8x1_sharedMemory_$1${numRegisterSuffix[d]}.csv
	    else 
	      output=./plotData/${precision}/Results_timing_${machines[m]}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_$1${numRegisterSuffix[d]}.csv
	    fi 
            
            #check if already exists
            if [ -f ${output} ]; then
	      echo WARNING: the ${output} already exists. Will be overritten! 
            fi	

	    # echo the csv header
	    echo "nTracks,time,time_low_error,time_high_error" > $output
	    for i in ${nTracks[@]}; do
	         if [ $1 -eq 1 ]; then
                   # Additional calculation of the gridSize
	           tracksPerGrid=`expr ${i} / ${k}`
	           gridSizeX=`echo ${gridSizes[j]} | sed 's/x1x1//g'`
	           echo tracksPerGrid=${tracksPerGrid} 
	           if [ ${tracksPerGrid} -gt ${gridSizeX} ]; then
	             gridSize=${tracksPerGrid}x1x1
	           else
	             gridSize=${gridSizes[j]}
                   fi		   
	           echo gridSize=${gridSize}
	           input=./results/test_launch_bounds/launch_bounds_${launchBounds[d]}/Results_timing_${machines[m]}_nTracks_${i}_nStreams_${k}_gridSize_${gridSize}_blockSize_8x8x1_sharedMemory_$1.csv
	           #input=./results/Results_timing_double_${machines[m]}_nTracks_${i}_nStreams_${k}_gridSize_${gridSize}_blockSize_8x8x1_sharedMemory_$1.csv
                 else
	           input=./results/test_launch_bounds/launch_bounds_${launchBounds[d]}/Results_timing_${machines[m]}_nTracks_${i}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_$1.csv
	           #input=./results/Results_timing_double_${machines[m]}_nTracks_${i}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_$1.csv
	         fi	

	    	 if [ ! -e ${input} ]; then 
	    	  echo ${input} does not exit!
	    	  exit
	         else
                   #check the lines of the input results
                   nTests=`wc -l < ${input}`
	           echo nTest = ${nTests}
                   if [ ${nTests} -ne 5 ]; then
                      echo WAENING: There are ${nTests} test results in ${input}. Are you sure about this?
                   fi
                 
	           mean=`getMean ${input}`
                   sigma=`getSigma ${input}`

                   # echo the ntracks, timing, timing_low_error, timing_high_error in csv format 
                   echo ${i}\,${mean}\,${sigma}\,${sigma} 
                   echo ${i}\,${mean}\,${sigma}\,${sigma} >> $output
	    	 fi	
	    done;
          done;
        done;
    done;
done;

