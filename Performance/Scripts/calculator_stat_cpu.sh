#!/bin/bash

RAW_DATA=../../externals/gpuKFPerformance/data/raw-data
TIMING=../../externals/gpuKFPerformance/data/timing

# check if the TIMING and RAW_DATA directory exist
if [ ! -d ${RAW_DATA} ]; then
  echo "Directory '${RAW_DATA}' DOES NOT exists."
  exit 1
fi
if [ ! -d ${TIMING} ]; then
   echo "Directory '${TIMING}' DOES NOT exists. Creating it." 
   mkdir ${TIMING}
fi

precision=float

#=======The Haswell test===================
machines=("Haswell_EigenInverter" "Haswell_customInverter")
machineso=("Haswell_EigenInverter" "Haswell_CustomInverter")
threads=(1 60)

#=========The KNL test====================
#machines=("Knl_EigenInverter" "Knl_customInverter")
#machineso=("KNL_EigenInverter" "KNL_CustomInverter")
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
        output=./${TIMING}/${precision}/Results_timing_${machineso[m]}_OMP_NumThreads_${threads[j]}.csv
        #check if already exists
        if [ -f ${output} ]; then
	  echo WARNING: the ${output} already exists. Will be overritten! 
        fi	

	# echo the csv header
	echo "nTracks,time,time_low_error,time_high_error" > $output
	for i in ${nTracks[@]}; do
	        input=./${RAW_DATA}/Results_timing_${machines[m]}_nTracks_${i}_OMP_NumThreads_${threads[j]}.csv
	        #input=./RAW_DATA/Results_timing_double_${machines[m]}_nTracks_${i}_OMP_NumThreads_${threads[j]}.csv
                
		if [ ! -f ${input} ]; then 
		  echo ${input} does not exit
		  exit
	        else
	          nTests=`wc -l < ${input}`	  
	          echo nTests = ${nTests}
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

