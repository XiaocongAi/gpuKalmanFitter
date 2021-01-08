#!/bin/bash

if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
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

machines=("v100")
nTracks=(5 10 50 100 500 1000 5000 10000) 
nStreams=(1 4) 
gridSizes=('20000*1*1')
blockSizes=('8*8*1')

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
	   if [ $1 -eq 1]; then 
	     output=./plotData/Results_timing_${machines[m]}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_8*8*1_sharedMemory_$1.csv
	   else 
	     output=./plotData/Results_timing_${machines[m]}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_$1.csv
	   fi 

	   for i in ${nTracks[@]}; do
	        if [ $1 -eq 1]; then 
	          input=./results/Results_timing_$1_nTracks_${i}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_8*8*1_sharedMemory_$1.csv
                else
	          input=./results/Results_timing_$1_nTracks_${i}_nStreams_${k}_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_$1.csv
		fi	

	   	if [ ! -f ${input} ]; then 
	   	  echo ${input} does not exit!
	   	  exit
	           else
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

