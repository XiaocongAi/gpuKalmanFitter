# gpuKalmanFitter

A R&D repository based on *A Common Tracking Software* [repository](https://github.com/acts-project/acts). 
It simplifies and transcribes the KalmanFitter in the ACTS repository to make it working on heterogeous computing.

Code dependency
---------------
* GCC compiler (min version 7.5.0)
* Nvidia CUDA (min version 10.2.89)
* Eigen library 
* CERN ROOT

The `gpu-kf-noRoot` branch contains the master code without the ROOT dependency.  
All the dependecies can be installed through [Spack](https://spack.readthedocs.io/en/latest/) package manager.

Build the code
--------------
`git clone https://github.com/XiaocongAi/gpuKalmanFitter.git`  
`cd gpuKalmanFitter`   
`cmake -S . -B <build_directory>`  
`cmake --build <build_directory> <options>`  
Invoke the executable:  
`./INSTALL/bin/KalmanFitterGPUTest -d gpu -t 1000 -g 5120x1 -b 8x8x1 -o 0`  

Running the executable in a container
-------------------------------------
A singularity container (1.04GB) with all the dependecies is made available to download either
* directly from the [cloud website](https://cloud.sylabs.io/library/hpc-uhh/default/gpu-kf), or 
* through singularity call: `singularity pull library://hpc-uhh/default/gpu-kf:v1.0`  

Currently, it runs the executable from tag [v2.0-noRoot](https://github.com/XiaocongAi/gpuKalmanFitter/tags). 

**Invocation examples**

To check the runtime options for the executable:  
`singularity run --nv gpu-kf_v1.0.sif --help`

To run the fitting for 10,000 tracks on your available Nvidia GPU, with default parameters:  
`singularity run --nv gpu-kf_v1.0.sif -d gpu -t 10000`
    
To run the fitting for 10,000 tracks on the CPU instead of the GPU:  
(Note that a CUDA driver and a CUDA runtime must be accessible for the executable to run!)  
`singularity run --nv gpu-kf_v1.0.sif -d cpu -t 10000 -a Intel_i6-5218`
    
To run the fitting for 10,000 tracks on multiple GPUs (if available):  
(Note that the implementation will distribute the workload among *all* reachable GPUs)  
`singularity run --nv gpu-kf_v1.0.sif -d gpu -t 10000 -u 1`
    
    
Developing the code
-------------------
Install the code dependecies listed below, fork a branch from master and submit a pull request to merge your changes at the end.
