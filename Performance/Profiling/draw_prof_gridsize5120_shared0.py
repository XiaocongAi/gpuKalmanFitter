import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch

fig, (ax1, ax2) = plt.subplots(2, 1, True, False)
#plt.yticks(rotation=30)

#CUDA_API_activities = ['cudaMalloc', 'cudaLaunchKernel', 'cudaDeviceSynchronize', 'cudaFree']

# subplot for nstreams=1 
start1=3274.43
ax1.broken_barh([(0, 3276.48-start1), (3276.54-start1, 14.509), (3291.09-start1, 64.222), (3355.31-start1, 3356.97-3355.31)], (5, 3), facecolors=('xkcd:red', 'goldenrod', 'wheat', 'pink'), hatch='')
ax1.broken_barh([(3276.51-start1, 3284.06-3276.51), (3291.05-start1, 57.632), (3348.68-start1, 3355.29-3348.68)], (1, 3),
               facecolors=('green', 'lightblue', 'purple'), hatch='')

ax1.set_ylim(0, 20)
ax1.set_xlim(0, 85)
#ax1.set_xlabel('Time (ms)')
ax1.set_yticks([2.5, 6.5])
ax1.set_yticklabels(['CUDA Stream', 'CUDA API\nActivities'])

# Legend
fakeredbar = mpatch.Rectangle((0, 0), 1., 1., fc="red")
fakegoldenrodbar = mpatch.Rectangle((0, 0), 1., 1, fc="goldenrod")
fakewheatbar = mpatch.Rectangle((0, 0), 1, 1, fc="wheat")
fakepinkbar = mpatch.Rectangle((0, 0), 1, 1, fc="pink")
fakegreenbar = mpatch.Rectangle((0, 0), 1, 1, fc="green")
fakelbluebar = mpatch.Rectangle((0, 0), 1, 1, fc="lightblue")
fakepurplebar = mpatch.Rectangle((0, 0), 1, 1, fc="purple")
ax1.legend([fakeredbar, fakegoldenrodbar, fakewheatbar, fakepinkbar, fakegreenbar, fakelbluebar, fakepurplebar], ['cudaMalloc', 'cudaLaunchKernel', 'cudaDeviceSynchronize', 'cudaFree', 'Memcpy HtoD', 'Kernel execution', 'Memcpy DtoH'], ncol=2)

# subplot for nstreams=4 
start2=3336.82
ax2.broken_barh([(0., 3338.71-start2), (3338.75-start2, 3346.5-3338.75), (3346.7-start2, 3408.49-3346.7), (3408.5-start2, 3410.14- 3408.5)], (17, 3), facecolors=('xkcd:red', 'goldenrod', 'wheat', 'pink'), hatch='')
ax2.broken_barh([(3338.72-start2, 3340.63-3338.72), (3346.49-start2, 56.111), (3402.6-start2, 3404.26-3402.6)], (13, 3), facecolors=('green', 'lightblue', 'purple'), hatch='')
ax2.broken_barh([(3347.49-start2, 3350.52-3347.49), (3350.53-start2, 46.909), (3397.45-start2, 3399.11-3397.45)], (9, 3), facecolors=('green', 'lightblue', 'purple'), hatch='')
ax2.broken_barh([(3351.74-start2, 3354.54-3351.74), (3354.61-start2, 50.534), (3405.15-start2, 3406.76-3405.15)], (5, 3), facecolors=('green', 'lightblue', 'purple'), hatch='')
ax2.broken_barh([(3354.76-start2, 3357.47-3354.76), (3357.55-start2, 48.313), (3406.82-start2, 3408.42-3406.82)], (1, 3), facecolors=('green', 'lightblue', 'purple'), hatch='')

ax2.set_ylim(0, 22)
ax2.set_xlim(0, 85)
ax2.set_xlabel('Time (ms)')
ax2.set_yticks([2.5, 6.5, 10.5, 14.5, 18.5])
ax2.set_yticklabels(['CUDA Stream 3', 'CUDA Stream 2', 'CUDA Stream 1', 'CUDA Stream 0', 'CUDA API\nActivities'])

fig.tight_layout()
plt.show()
