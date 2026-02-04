# Nsight System

## Profile

```bash
CUDA_VISIBLE_DEVICES=1 nsys profile /path/to/executable ARGS...
```

## See Dumped Stats

### Overall Summary

```bash
nsys stats report1.nsys-rep
```

### CUDA API Statistics

```bash
nsys stats --report cuda_api_sum report1.nsys-rep
```

### CUDA GPU Kernel Statistics

```bash
nsys stats --report cuda_gpu_kern_sum report1.nsys-rep
```

### CUDA Memory Operations

```bash
nsys stats --report cuda_gpu_mem_sum report1.nsys-rep
```

### NVTX Ranges (If You Used NVTX Markers)

```bash
nsys stats --report nvtx_sum report1.nsys-rep
```

