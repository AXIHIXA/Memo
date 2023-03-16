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

#### 📌 Latency Hiding

- *Latency*
  - Number of clock cycles between an instruction being issued and being completed
- *Latency Hiding*
  - Latency of each instruction can be hidden by issuing other instructions in other resident warps
  - CPU cores are designed to minimize latency
  - GPU cores are designed to maximize throughput
    - GPU instruction latency is hidden by computation from other warps
- Instruction can be classifed into two basic types
  - Arithmetic instructions
    - Latency 10-20 cycles for arithmetic operations
  - Memory instructions
    - Latency 400-800 cycles for global memory accesses
- *Required Parallelism*
  - Arithmetic instructions: Number of required warps to hide latency
    - *Little's Law*: Required parallelism = Latency $\times$ Throughput
    - Two ways to increase parallelism
      - **Instruction-level parallelism (ILP)**: More independent instructions within a thread
      - **Thread-level parallelism (TLP)**: More concurrently eligible threads
  - Memory instructions: Number of Bytes of memory load/store per cycle required to hide latency
    - Check device's memory frequency via `$ nvidia-smi -a -q -d CLOCK | fgrep -A 3 "Max Clocks" | fgrep "Memory"`
      - E.g. NVIDIA Geforce GTX 2080 Ti: 7000 MHz (cycle per second)
      - Bandwidth (GB/s) $\div$ Memory frequency (cycle/s) => Bytes/cycle

#### 📌 Occupancy

- *Occupancy*
  - Ratio of active warps to maximum number of warps per SM
  - Check the maximum warps per SM for your device
    ```c++
    cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * prop, int device);
    ```
  - Maximum number of threads per SM is returned in `prop->maxThreadsPerMultiProcessor`
  - Manipulating thread blocks to either extreme can restrict resource utilization
    - **Small thread blocks**
      - Limits the number of warps per SM to be reached before all resources are fully utilized
    - **Large thread blocks**
      - Leads to fewer per-SM hardware resources available to each thread
- Guidelines for grid and block size
  - Keep the number of threads per block a multiple of warp size 32
  - Avoid small block sizes: Start with at least 128 or 256 threads per block
  - Adjust block size up or down according to kernel resource requirements
  - Keep the number of blocks much greater than the number of SMs to expose sufficient parallelism to your device
  - COnduct experiments to discover the best execution configuration and resource usage

#### 📌 Synchronization

- Two levels of Barrier synchronoizations in CUDA
  - **System-level**
    - Wait for all work on both the host and the device to complete 
      - Many CUDA API calls and all kernel launches are asynchronous w.r.t. the host
      ```c++
      /// May return errors from previous async CUDA operations
      cudaError_t cudaDeviceSynchronize(void);
      ```
  - **Block-level**
    - Wait for all threads in a thread block to reach the same point in execution on the device
      - Warps in a thread block are executed in an undefined order
      ```c++
      /// Each thread in the same thread block must wait 
      /// until all other threads in that block have reached this synchronization point. 
      /// All global and shared memory accesses made by all threads prior to this barrier
      /// will be visible to all other threads in the thread block after the barrier. 
      __device__ void __syncthreads(void);
      ```
    - **No** thread synchronization among different blocks
      - The only safe way to synchronize across blocks is 
        to use the global synchronization point at the end of every kernel execution

### 🎯 EXPOSING PARALLELISM

- Metrics and performace
  - **No** single metric can prescribe optimal performance.
   - Which metric or event most directly relates to overall performance 
     depends on the nature of the kernel code.
  - Seek a good balance among related metrics and events.
  - Check the kernel from different angles to find a balance among the related metrics.
  - Grid/block heuristics provide a good starting point for performance tuning. 
- Practice *grid and block heuristics* with the following code
  - Higher occupancy $\ne$ higher performace
  - Higher load throughput $\ne$ higher performance
  - More thread blocks launched $\ne$ higher performance
```c++
__global__ void sumMatrixOnGPU2D(float * A, float * B, float * C, int NX, int NY) 
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY) 
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char * argv[])
{
    int nx = 1 << 14, ny = 1 << 14, dimx, dimy;

    if (2 < argc)
    {
        dimx = std::atoi(argv[1]);
        dimy = std::atoi(argv[2]);
    }

    dim3 blockDim {dimx, dimy};
    dim3 gridDim {(nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y};

    ...
}
```

#### 📌 Checking Active Warps with `nvprof`

```bash
$ nvcc -O3 --generate-code=arch=compute_75,code=[compute_75,sm_75] sumMatrix.cu -o sumMatrix

$ ./sumMatrix 32 32
sumMatrixOnGPU2D <<< (512,512), (32,32) >>> elapsed 60 ms
$ ./sumMatrix 32 16
sumMatrixOnGPU2D <<< (512,1024), (32,16) >>> elapsed 38 ms
$ ./sumMatrix 16 32
sumMatrixOnGPU2D <<< (1024,512), (16,32) >>> elapsed 51 ms
$ ./sumMatrix 16 16
sumMatrixOnGPU2D <<< (1024,1024),(16,16) >>> elapsed 46 ms

$ nvprof --metrics achieved_occupancy ./sumMatrix 32 32
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> Achieved Occupancy 0.501071
$ nvprof --metrics achieved_occupancy ./sumMatrix 32 16
sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> Achieved Occupancy 0.736900
$ nvprof --metrics achieved_occupancy ./sumMatrix 16 32
sumMatrixOnGPU2D <<<(1024,512), (16,32)>>> Achieved Occupancy 0.766037
$ nvprof --metrics achieved_occupancy ./sumMatrix 16 16
sumMatrixOnGPU2D <<<(1024,1024),(16,16)>>> Achieved Occupancy 0.810691
```
- Because the second case has more blocks than the first case, 
  it exposed more active warps to the device.
  - This is likely the reason why the second case has a higher achieved occupancy 
    and better performance than the first case.
- The fourth case has the highest achieved occupancy, but it is **not** the fastest. 
  - Therefore, a higher occupancy does **not** always equate to higher performance. 
  - There must be other factors that restrict performance.

#### 📌 Checking Active Warps with `nvprof`

- `C[idx] = A[idx] + B[idx]` has two loads and one store
```bash
$ nvprof --metrics gld_throughput./sumMatrix 32 32
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> Global Load Throughput 35.908GB/s
$ nvprof --metrics gld_throughput./sumMatrix 32 16
sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> Global Load Throughput 56.478GB/s
$ nvprof --metrics gld_throughput./sumMatrix 16 32
sumMatrixOnGPU2D <<<(1024,512), (16,32)>>> Global Load Throughput 85.195GB/s
$ nvprof --metrics gld_throughput./sumMatrix 16 16
sumMatrixOnGPU2D <<<(1024,1024),(16,16)>>> Global Load Throughput 94.708GB/s
```
- While the fourth case has the highest load throughput, 
  it is slower than the second case (which only demonstrates around half the load throughput). 
  - A higher load throughput does **not** always equate to higher performance. 
```bash
# The ratio of requested global load throughput to required global load throughput
$ nvprof --metrics gld_efficiency ./sumMatrix 32 32
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> Global Memory Load Efficiency 100.00%
$ nvprof --metrics gld_efficiency ./sumMatrix 32 16
sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> Global Memory Load Efficiency 100.00%
$ nvprof --metrics gld_efficiency ./sumMatrix 16 32
sumMatrixOnGPU2D <<<(1024,512), (16,32)>>> Global Memory Load Efficiency 49.96%
$ nvprof --metrics gld_efficiency ./sumMatrix 16 16
sumMatrixOnGPU2D <<<(1024,1024),(16,16)>>> Global Memory Load Efficiency 49.80%
```
- The load efficiency for the last two cases was half that of the first two cases. 
  - This would explain why the higher load throughput and achieved occupancy 
    of the last two cases did not yield improved performance. 
  - Even though the number of loads being performed is greater for the last two cases, 
    the effectiveness of those loads is lower.
- Note that the common feature for the last two cases is that their block size 
  in the innermost dimension is half of a warp. 
  - **The innermost dimension (`blockDim.x`) should always be a multiple of the warp size**. 

### 🎯 AVOIDING BRANCH DIVERGENCE






## 🌱 

### 🎯 

#### 📌 

## 🌱 

### 🎯 

#### 📌 
