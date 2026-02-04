# Nsight System

## Profile

CUDA_VISIBLE_DEVICES=1 nsys profile /path/to-executable ARGS...

## See Dumped Stats

### Overall summary

nsys stats report1.nsys-rep

### CUDA API statistics

nsys stats --report cuda_api_sum report1.nsys-rep

### CUDA GPU kernel statistics

nsys stats --report cuda_gpu_kern_sum report1.nsys-rep

### CUDA memory operations

nsys stats --report cuda_gpu_mem_sum report1.nsys-rep

### NVTX ranges (if you used NVTX markers)

nsys stats --report nvtx_sum report1.nsys-rep
