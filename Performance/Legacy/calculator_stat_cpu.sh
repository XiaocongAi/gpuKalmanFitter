#!/bin/bash

# check if the plotData and results directory exist
if [ ! -d "results" ]; then
  echo "Directory 'results' DOES NOT exists."
  exit 1
fi
if [ ! -d "plotData" ]; then
   echo "Directory 'plotData' DOES NOT exists. Creating the 'plotData' directory." 
   mkdir plotData
fi

machines=("Haswell_EigenInverter" "Haswell_customInverter")
threads=(1 60)
#machines=("Knl_EigenInverter" "Knl_customInverter")
#threads=(1 250)

nTracks=(5 10 50 100 500 1000 5000 10000 50000 100000) 

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
    for ((j=0; j<${#threads[@]};++j)); do
        output=./plotData/double/Results_timing_${machines[m]}_OMP_NumThreads_${threads[j]}.csv
        #check if already exists
        if [ -f ${output} ]; then
	  echo WARNING: the ${output} already exists. Will be overritten! 
        fi	

	# echo the csv header
	echo "nTracks,time,time_low_error,time_high_error" > $output
	for i in ${nTracks[@]}; do
	        input=./results/Results_timing_double_${machines[m]}_nTracks_${i}_OMP_NumThreads_${threads[j]}.csv
                
		if [ ! -f ${input} ]; then 
		  echo ${input} does not exit
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

