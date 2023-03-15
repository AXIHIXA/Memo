# Professional CUDA C Programming Notes



## 🌱 1 Heterogeneous Parallel Computing with CUDA

### 🎯 PARALLEL COMPUTING -- Computer Architecture

- *Flynn's Taxonomy*
  - Single Instruction Single Data (SISD)
    - Traditional computer
  - Single Instruction Multiple Data (SIMD)
    - Muliple cores
    - All cores execute the same instruction at any time
    - Each operate on different data stream
  - Multiple Instruction Single Data (MISD)
    - Uncommon
    - E.g., vector computers
  - Multiple Instruction Multiple Data (MIMD)
- Objectives
  - Latency: Time take for an operaation to staart and complete
    - Time to complete an operation
    - microseconds
  - Bandwidth: Amount of data can be processed per unit time
    - MB/s, GB/s
  - Throughput: Amount of operations that can be processed per unit time
    - Number of operations processed in a given time unit
    - *gflops* (billion floating-point operaions per second)
- Memory organization
  - Multi-node with distributed memory
    - Many processers connected by a network
    - Each processor has its own local memory
    - Processers can communicate their local memories over the network
    - Referred as *clusters*
  - Multiprocesor with shared memory
    - Processors are 
      - either physically connected to the same memory
      - or share a low-latency link, e.g., PCI-Express (PCIe)
    - Same address space (there could be multiple physical memories)
- NVIDIA: *Single Instruction, Multiple Thread* (SIMT)

### 🎯 HETEROGENEOUS COMPUTING

- Performace features
  - *Peak computational performance* 
    - How many single-precision or double-precision floating point calculations can be processed per second
    - `gflops` (billion) or `tflops` (trillion)
  - *Memory bandwidth*
    - Ratio at which data could be read from or stored to memory
    - GB/s
- CPU Threads
  - Heavyweight
  - Context switches are slow and expensive
- GPU Threads
  - Lightweight
  - Thousands of threads queued up for work



## 🌱 2 CUDA Programming Model

### 📌 [Managing Memory](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

```c++
/// Allocate memory on the device. 
/// @return cudaError_t typed value, takes one of the following values:
///         - cudaSuccess 
///         - cudaErrorMemoryAllocation
__host__​ __device__ cudaError_t cudaMalloc(void ** devPtr, size_t size);

/// Copies data between host and device.
/// @param kind takes one of the following types:
///             - cudaMemcpyHostToHost
///             - cudaMemcpyHostToDevice
///             - cudaMemcpyDeviceToHost
///             - cudaMemcpyDeviceToDevice
__host__ cudaError_t cudaMemcpy(void * dst, void * src, size_t count, cudaMemcpyKind kind);

/// Initializes or sets device memory to a value.
__host__ cudaError_t cudaMemset(void * devPtr, int value, size_t count);

/// Frees memory on the device.
__host__ ​__device__ cudaError_t cudaFree(void * devPtr);

/// Convert an error code to human-readable error message. 
/// Returns the description string for an error code.
__host__ ​__device__ ​const char * cudaGetErrorString(cudaError_t error);
```
- GPU Memory Hierachy
  - Global memory: analogous to CPU system memory
  - Shared memory: analogous to CPU cache, but could be directly controlled from a CUDA C kernel
- Different Memory Spaces
  - **Never** ~~dereference the different memory spaces~~
  - E.g., `gpuRef = devPtr` instead of `cudaMemcpy(gpuRef, devPtr, nBytes, cudaMemcpyDeviceToHost)` will crash

## 📌 Organizing Threads

- Grid
  - All threads spawned by a single kernel launch are collectively called a *grid*. 
    - 1D, 2D, or 3D, `gridDim.xyz`, `blockIdx.xyz`
  - All these grids share the same global memory space. 
  - A grid is made up of multiple thread blocks. 
- Thread block
  - 1D, 2D, or 3D, `blockDim.xyz`, `threadIdx.xyz`
  - One thread block ~ One Streaming Multiprocessor (SM)
  - A group of threads that can cooperate with each other using
    - Block-local synchronization
    - Block-local shared memory
  - Threads from different blocks can **not** cooperate
- Grid and block dimensions
  - (Usually) 
    - Grids are organized as 2D arrays of blocks
    - Blocks are organized as 3D arrays of threads
  - `dim3`
    - Both grids and blocks use the `dim3` type with 3 unsinged integer fields. 
    - Ununsed fields with be initialized to 1 and ignored. 

## 📌 Launching a CUDA Kernel

- All CUDA kernel launches are asynchronous.
  - Control returns to CPU immediately after the CUDA kernel is invoked. 
```c++
dim3 gridDim(...), blockDim(...);
cudaKernelFunc<<<gridDim, blockDim>>>(arguments);
```

## 📌 Writing Your Kernel

- Function type qualifiers
  - `__global__`: Functions defined with `__global__` are kernel functions
    - Executed on the device
    - Callable from
      - the host
      - the device
    - Must have a `void` return type
  - `__device__`
    - Executed on the device
    - Callable from the device only
  - `__host__`
    - Executed on the host
    - Callable from the host only
    - Can be omitted
- CUDA kernels are functions with restrictions
  - Access to device memory only
  - Must have `void` return type
  - **No** support for
    - ~~A variable number of arguments~~
    - ~~Static variables~~
    - ~~Function points~~
    - Exhibit an asynchronous behavior

## 📌 Verifying Your Kernel

- You can use `printf` in your kernel for Fermi and later generation devices;
- You can set the execution configuration to `<<<1, 1>>>` to force a sequential implementation. 

## 📌 Timing Your Kernel

- With CPU timer
- With `nvprof`
  - Command-line profiling tool



## 🌱 3 CUDA Execution Model

### 🎯 INTRODUCTION

#### 📌 GPU Architecture Overview

- GPU architecture is built around a scalable array of *Streaming Multiprocessers* (SM)
  - Each SM is designed to support concurrent execution of hundreds of threads
    - Each thread block is assigned to one SM
      - Threads of this thread block execute concurrently only on this SM
    - Multiple thread blocks may be assigned on the same SM
    - Instructions within a single thread are pipelined to further leverage instruction-level parallelism
- Key components of a Fermi SM
  - CUDA Cores
  - Shared Memory / L1 Cache
  - Register File
  - Load/Store Units
  - Special Function Units
  - Warp Scheduler
- CUDA employs a SIMT architecture 
  - Execute threads in groups of 32 called *warps*
  - All threads in a warp execute the same instruction at the same time
    - Each thread has its own instruction address counter and register state
    - Each thread carries out the current instruction on its own data
  - Each SM partitions its assigned thread block into warps of 32 threads
- SIMT vs SIMD
  - Both broadcast the same instruction to multiple execution units
  - Difference
    - SIMD requires all vector elements in a vector execute together in a unified synchronous group
    - SIMT allows multiple threads in the same warp to execute independently
      - All threads in a warp start together at the same program address
      - Indivisual threads could have different behavior
  - SIMT includes three key features (that SIMD does not)
    - Each thread has its own instruction address counter
    - Each thread has its own register state
    - Each thread can have an independent execution path
- 




## 🌱 

### 🎯 

#### 📌 

## 🌱 

### 🎯 

#### 📌 
