#!/bin/bash


nTracks=(5 10 50 100 500 1000 5000 10000) 
gridSizes=('20000*1*1')
blockSizes=('8*8*1')

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

for ((j=0; j<${#gridSizes[@]};++j)); do
        output=Results_timing_gpu_nStreams_1_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_0.csv
  
	for i in ${nTracks[@]}; do
	        fileName=Results_timing_gpu_nTracks_${i}_nStreams_1_gridSize_${gridSizes[j]}_blockSize_${blockSizes[j]}_sharedMemory_0.csv
                
		if [ ! -f ${fileName} ]; then 
		  echo ${fileName} does not exit
		  exit
	        else
		  sum=`perl -lne '$x += $_; END { print $x; }' < ${fileName}`
	          numLine=`wc -l < ${fileName}`	  
	          average=`echo "${sum} / ${numLine} " | bc -l` 
		  echo sum=${sum}, average=${average} for tracks ${i} 
		  min=99999
		  max=0
		  echo=`printf ${fileName} | sort | head ${fileName}` 
		  while IFS= read -r line
                    do
		      newline=`echo ${line} | sed 's/,//g'` 
                      echo "$newline"
		      getMax ${max} ${newline} 
		      getMin ${min} ${newline} 
	          done < "${fileName}"
		  echo min=${min}, max=${max}
		  lowErr=`echo "${average} - ${min}" |bc -l`
		  upErr=`echo "${max} - ${average}" |bc -l`
		  echo ${i}, ${average} - ${lowErr} + ${upErr} >> $output 
		fi	
	done;
done;

