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

machines=("Tesla_V100-SXM2-16GB")
nTracks=(5 10 50 100 500 1000 5000 10000 50000 100000) 
nStreams=(1 4)
#Note the script could only handle the real 1D gridSize
gridSizes=('100000x1x1' '5120x1x1')
blockSizes=('8x8x1' '8x8x1')

### griSizes and blockSizes list for other blockSizes configurations ###
# gridSizes=('100000x1x1' '100000x1x1' '100000x1x1' '100000x1x1' '100000x1x1' '5120x1x1' '5120x1x1' '5120x1x1' '5120x1x1' '5120x1x1')
# blockSizes=('16x16x1' '32x32x1' '64x1x1' '256x1x1' '1024x1x1' '16x16x1' '32x32x1' '64x1x1' '256x1x1' '1024x1x1')
############################################################


# helper functions to compare two floats
getMax(){
if (( $(echo "$1 > $2" |bc -l) )); then
  max=$1
 else 
  max=$2
fi
}
getMin(){
if (( $(echo "$1 < $2" |bc -l) )); then
  min=$1
 else 
  min=$2
fi
}

for ((m=0; m<${#machines[@]};++m)); do
    for ((j=0; j<${#gridSizes[@]};++j)); do
        for k in ${nStreams[@]}; do
	   if [ $1 -eq 1 ]; then 
	     output=./plotData/Results_timing_${machines[m]}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_8*8*1_sharedMemory_$1.csv
	   else 
	     output=./plotData/Results_timing_${machines[m]}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_$1.csv
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
		  gridSizeX=`echo ${gridSizes[j]} | sed 's/*1*1//g'`
		  echo tracksPerGrid=${tracksPerGrid} 
		  if [ ${tracksPerGrid} -gt ${gridSizeX} ]; then
		    gridSize=${tracksPerGrid}\*1\*1
		  else
		    gridSize=${gridSizes[j]}
                  fi		   
		  echo gridSize=${gridSize} 
	          input=./results/Results_timing_${machines[m]}_nTracks_${i}_nStreams_${k}_gridSize_${gridSize}_blockSize_8\*8\*1_sharedMemory_$1.csv
                else
	          input=./results/Results_timing_${machines[m]}_nTracks_${i}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_$1.csv
		fi	

	   	if [ ! -e ${input} ]; then 
	   	  echo ${input} does not exit!
	   	  exit
	        else
                  #check the lines of the input results
                  nTests=`wc -l < ${input}`
		  echo nTest = ${nTests}
                  if [ ${nTests} -ne 5 ]; then
                     echo WAENING: There are ${nTests} test results for one point. Are you sure about this?
                  fi

	   	  sum=`perl -lne '$x += $_; END { print $x; }' < ${input}`
	             numLine=`wc -l < ${input}`	  
	             average=`echo "scale=2; ${sum} / ${numLine} " | bc -l` 
	   	  echo sum=${sum}, average=${average} for tracks ${i} 
	   	  
	   	  min=99999
	   	  max=0
	   	  while IFS= read -r line
                       do
	   	      newline=`echo ${line} | sed 's/,//g'` 
                         echo "$newline"
	   	      getMax ${max} ${newline} 
	   	      getMin ${min} ${newline} 
	             done < "${input}"
	   	  echo min=${min}, max=${max}
	   	  lowErr=`echo "scale=2; ${average} - ${min}" |bc -l`
	   	  highErr=`echo "scale=2; ${max} - ${average}" |bc -l`
	   	 
	   	  # echo the ntracks, timing, timing_low_error, timing_high_error in csv format 
	   	  echo ${i}\,${average}\,${lowErr}\,${highErr} >> $output 
	   	fi	
	   done;

        done;
    done;
done;

