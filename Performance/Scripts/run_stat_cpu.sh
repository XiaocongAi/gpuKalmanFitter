#!/bin/bash

if [ $# -ne 2 ]; then
  echo 1>&2 "Usage: $0 <EXECUTABLE> <MACHINE>, e.g. $0 <build>/Run/CPU/KalmanFitterCPUTest Intel_i7-8559U_EigenInverter"
  exit 3
fi

#The input parameter is the location of the executable and the machine name
echo "Start statistical runs for $1 $2"

nTracks=(5 10 50 100 500 1000 5000 10000 50000 100000) 

#Threads on Cori Haswell
nThreads=(1 60)

#Threads on Cori KNL 
#nThreads=(1 250)


#10 measurements per point
nTestsPerMeasurement=10

for i in ${nTracks[@]}; do
       for j in ${nThreads[@]}; do
		for cnt in {1..${nTestsPerMeasurement}}; do
                	echo "Run $cnt for ${i} tracks"
                	$1 -t ${i} -o 0 -a $2 -r ${j} 
                	sleep 1;
		done;
       done;
done;
