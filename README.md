# gpuKalmanFitter

A R&D repository based on *A Common Tracking Software* [repository](https://github.com/acts-project/acts). 
It simplifies and transcribes the KalmanFitter in the ACTS repository to make it working on heterogeous computing.

Code dependency
---------------
* GCC compiler (min version 7.5.0)
* Nvidia CUDA (min version 10.2.89)
* Eigen library 
* (optional) CERN ROOT 

All the dependecies can be installed through [Spack](https://spack.readthedocs.io/en/latest/) package manager.

Running the code
----------------
A singularity container (1.2GB) with all the dependecies is made available to download either
* directly from the [cloud website](https://cloud.sylabs.io/library/_container/60656f4165dbc33da1911e37), or 
* through singularity call:  
`singularity pull library://hpc-uhh/default/gpukf:cuda`.   

Currently, it runs the executable from tag [v1.1-noRoot](https://github.com/XiaocongAi/gpuKalmanFitter/tags). 

**Invocation examples**

To check the runtime options for the executable:  
`singularity run --nv gpukf_cuda.sif --help`

To run the fitting for 10,000 tracks on your available Nvidia GPU, with default parameters:  
`singularity run --nv gpukf_cuda.sif -d gpu -t 10000`
    
To run the fitting for 10,000 tracks on the CPU instead of the GPU:  
`singularity run --nv gpukf_cuda.sif -d cpu -t 10000 -a Intel_i6-5218`
    
To run the fitting for 10,000 tracks on multiple GPUs (if available):  
`singularity run --nv gpukf_cuda.sif -d gpu -t 10000 -u 1`
    
    
Developing the code
-------------------
Install the code dependecies listed below, fork a branch from master and submit a pull request to merge your changes at the end.
