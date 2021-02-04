import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
# matplotlib inline

# Data
cudaMallocHost = 2.8818098e+2 
cudaDeviceSynchronize = 63.931147
cudaLaunchKernel = 14.402251
cudaFreeHost = 13.498389
cudaMalloc = 1.739605
cudaFree = 1.562450
cudaMemcpyAsync = 0.049472
cudaMemcpy = 0.048701
cudaStreamCreate = 0.042855
cudaStreamDestroy = 0.007433 
memcpyHtoD = 7.506199
memcpyDtoH = 6.598752
kernel = 57.341364

# Create arrays for the plot
#activities = ['cudaMallocHost', 'cudaDeviceSynchronize', 'cudaLaunchKernel', 'cudaFreeHost', 'cudaMalloc', 'cudaFree', 'cudaMemcpyAsync', 'cudaMemcpy', 'cudaStreamCreate', 'cudaStreamDestroy']
activities = ['cudaMallocHost', 'cudaMalloc', 'cudaLaunchKernel', 'cudaDeviceSynchronize', 'cudaFree', 'cudaFreeHost']
x_pos = np.arange(len(activities))
#acti_time = [cudaMallocHost, cudaDeviceSynchronize, cudaLaunchKernel, cudaFreeHost, cudaMalloc, cudaFree, cudaMemcpyAsync, cudaMemcpy, cudaStreamCreate, cudaStreamDestroy]
acti_time = [cudaMallocHost, cudaMalloc, cudaLaunchKernel, cudaDeviceSynchronize, cudaFree, cudaFreeHost]
memcpyHtoD_time = [0, 0, memcpyHtoD, 0, 0, 0]
memcpyDtoH_time = [0, 0, 0, memcpyDtoH, 0, 0]
kernel_time = [0, 0, 0, kernel, 0,  0]

# Build the plot
fig, ax = plt.subplots()
plt.xticks(rotation=30)

bar1 = ax.bar(x_pos, acti_time, align='center', alpha=0.5)
bar2 =  ax.bar(x_pos, memcpyHtoD_time, align='center', alpha=0.5)
bar3 = ax.bar(x_pos, memcpyDtoH_time, align='center', alpha=0.5)
bar4 = ax.bar(x_pos, kernel_time, align='center', alpha=0.5)
ax.set_ylabel('Time (ms)')
ax.set_xticks(x_pos)
ax.set_xticklabels(activities)
#ax.set_title('CUDA API activities')
#ax.yaxis.grid(True)

ax.legend((bar1, bar2, bar3, bar4), ('CUDA API activities', 'CUDA memcpy HtoD', 'CUDA memcpy DtoH', 'CUDA Kernel'))



# Save the figure and show
plt.tight_layout()
plt.savefig('cuda_API_activities.png')
plt.show()
