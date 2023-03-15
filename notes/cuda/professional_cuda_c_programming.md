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

### 📌 Organizing Threads

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

### 📌 Launching a CUDA Kernel

- All CUDA kernel launches are asynchronous.
  - Control returns to CPU immediately after the CUDA kernel is invoked. 
```c++
dim3 gridDim(...), blockDim(...);
cudaKernelFunc<<<gridDim, blockDim>>>(arguments);
```

### 📌 Writing Your Kernel

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

### 📌 Verifying Your Kernel

- You can use `printf` in your kernel for Fermi and later generation devices;
- You can set the execution configuration to `<<<1, 1>>>` to force a sequential implementation. 

### 📌 Timing Your Kernel

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
    - Each SM can hold more than one thread block at the same time
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
      - All threads in a warp start together (logically) at the same program address
        - Physically, not all threads can execute at the same time
      - Different threads may make progress at a different pace 
  - SIMT includes three key features (that SIMD does not)
    - Each thread has its own instruction address counter
    - Each thread has its own register state
    - Each thread can have an independent execution path
- Magic Number 32
  - Optimizing workloads to fit within the boundaries of a warp (group of 32 threads) 
    will lead to more efficient utilization of GPU compute resources
- Shared Memory and Registers
  - Shared memory is partitioned among thread blocks
  - Registers are partitioned among threads
  - Thread in a thread block can cooperate and communicate through these resources
- Data race
  - CUDA provides means to synchronize threads within a thread block
    - to ensure that all thread reach certain points in execution before making further progresses
  - **No** inter-block synchronization
- Warp context switching
  - When a warp idles (e.g., waiting for values to be read from device memory)
    the SM is free to schedule another available warp from any assigned thread block 
  - **No** overhead
    - Hardware resources are partitioned among all threads and blocks on an SM
    - State of the newly scheduled warp is already stored on the SM
- *Dynamic Parallelism*
  - Any kernel can launch another kernel and manage any inter-kernel dependencies
  - Aids recursive and data-dependent execution patterns

#### 📌 Profile-Driven Optimization

- Profiling is important in CUDA programming because
  - A naive implementation generally does **not** yield best performance. 
    - Profiling tools can help you find the bottleneck regions of your code
  - CUDA partitions the compute resources in an SM among multiple thread blocks
    - This partition causes some resources to become performance limiters
    - Profiling tools can help you gain insight into how compute resources are being utilized
  - CUDA provides an abstraction of the hardware architecture enabling you to control thread concurrency
    - Profiling tools can help you measure, visualize, and guide your implementation
- CUDA provides two profiling tools
  - `nvvp`: Standalone visual profiler
    - Displays a timeline of program activity on both the CPU and the GPU
    - Analyzes for potential bottlenecks and suggests how to eliminate/reduce them
  - `nvprof`: Command-line profiler
- Events and metrics
  - An event is a countable activity the corresponds to a hardaware counter collected during kernel execution
  - A metric is a characteristic of a kernel calculated from one or more events
  - Most counters are reported per SM **rather than** the entire GPU
  - A single run can only collect a few counters
    - The collection of some counters is mutually exclusive
    - Multiple profiling runs are often needed to gather all relevant counters
  - Counter values may **not** be exactly the same across repeated runs
    - Due to variations in GPU execution (E.g., thread block and warp scheduling order)

### 🎯 WARP EXECUTION

#### 📌 Warps and Thread Blocks

- When you launch a grid of thread blocks
  - These thread blocks are distributed among SMs
  - Once a thread block is assigned to a SM
    - Threads in this thread block are further partitioned into warps
    - A warp consists of 32 consecutive threads
    - All threads in a warp are executed in SIMT fashion
      - All threads execute the same instruction
      - Each thread carries out that operation on its own private data
- The hardware always allocates a discrete number of warps for a thread block
  - `numberOfWarps = ceil(numberOfThreads / warpSize)`
  - A warp is **never** split between different thread blocks
  - If thread block size is not a multiple of warp size, some threads in the last warp a left inactive
    - But they still consume SM resources, e.g., CUDA cores and registers!

#### 📌 Warp Divergence

- GPU has **no** complex branch prediction mechanism
  - All threads in a warp must execute the same instruction
- Threads in the saem warp executing different instructions is referred as *warp divergence*
  - When threads in a warp diverge, different `if-then-else` branches are executed serially
    - Warp serially executes each branch path 
    - disabling threads that do not take that path
  - Warp divergence can cause degraded performance
- Branch divergence only occurs within a warp
  - Different conditional values in different warps do **not** cause warp diverence
- E.g. For a 1D thread block
  - `if (threadIdx.x % 2 == 0)` 
    - even-numbered threads take `if` clause 
    - odd-numbered threads take `else` clause
  - Interleave data using a warp approach
    - `if ((threadIdx.x / warpSize) % 2 == 0)`
    - Forces the branch granularity to be a multiple of warp size
    - even warps take the `if` clause
    - odd warps the the `else` clause
- *Branch Efficiency*
  - The ratio of non-divergent branches to total branches
- Compiler optimizations
  - In branch prediction, a predicate variable for each thread is replaced by 1 or 0
  - Both conditional flow paths are fully executed
    - Only instructions with a predicate of 1 are executed
    - Instructions with 0 predicates are **not** executed
      - Corresponding thread does **not** stall either
  - Compiler replaces a branch instruction with predicated instructions 
    - only if the number of instructions in the body of a conditional statement is less than a certain threshold
  - A long code path will certainly result in warp divergence
```c++
__global__ void goodPractice(float * c)
{
    float a = 0.0f, b = 0.0f;

    if ((threadIdx.x / warpSize) % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }

    c[threadIdx.x] = a + b;
}

__global__ void doesNotDivergeThanksToCompilerOptimizations(float * c)
{
    float a = 0.0f, b = 0.0f;

    if (threadIdx.x % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }

    c[threadIdx.x] = a + b;
}

__global__ void willCertainlyDiverge(float * c)
{
    float a = 0.0f, b = 0.0f;
    bool pred = threadIdx.x % 2 == 0;

    if (pred)
    {
        a = 100.0f;
    }

    if (!pred)
    {
        b = 200.0f;
    }

    c[threadIdx.x] = a + b;
}
```

#### 📌 Resource Partitioning

- Local execution context of a warp mainly consists of
  - Program counters
  - Registers
  - Shared memory
- Switching from one execution context to another has no cost
- A thread block is called an *active block* when compute resources have been allocated to it
  - The warps it contains are called *active warps*
  - Active warps can be classified into
    - Selected warp: Warp that is actively executing
    - Stalled warp: Not ready for execution
    - Eligible warp: Ready for execution but not currently executing
      - A warp is elligible if both
        - 32 CUDA cores are available for execution, and
        - All arguments to the current instruction are ready

#### Latency Hiding




## 🌱 

### 🎯 

#### 📌 

## 🌱 

### 🎯 

#### 📌 
