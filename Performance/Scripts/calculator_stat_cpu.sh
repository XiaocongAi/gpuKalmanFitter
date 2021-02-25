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

precision=float

machines=("Haswell_customInverter")
threads=(60)
#machines=("Knl_EigenInverter" "Knl_customInverter")
#threads=(1 250)

nTracks=(5 10 50 100 500 1000 5000 10000 50000 100000) 

# helper functions to get mean and sigma 
getMean(){
awk 'BEGIN{s=0;}{s=s+$1;}END{print s/NR;}' $1
}

getSigma(){
awk '{delta = $1 - avg; avg += delta / NR; mean2 += delta * ($1 - avg); } END { print sqrt(mean2 / NR); }' $1
}


for ((m=0; m<${#machines[@]};++m)); do
    for ((j=0; j<${#threads[@]};++j)); do
        output=./plotData/${precision}/Results_timing_${machines[m]}_OMP_NumThreads_${threads[j]}.csv
        #check if already exists
        if [ -f ${output} ]; then
	  echo WARNING: the ${output} already exists. Will be overritten! 
        fi	

	# echo the csv header
	echo "nTracks,time,time_low_error,time_high_error" > $output
	for i in ${nTracks[@]}; do
	        input=./results/Results_timing_${machines[m]}_nTracks_${i}_OMP_NumThreads_${threads[j]}.csv
	        #input=./results/Results_timing_double_${machines[m]}_nTracks_${i}_OMP_NumThreads_${threads[j]}.csv
                
		if [ ! -f ${input} ]; then 
		  echo ${input} does not exit
		  exit
	        else
	          nTest=`wc -l < ${input}`	  
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

