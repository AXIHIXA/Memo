# Professional CUDA C Programming Notes



## 🌱 1 Heterogeneous Parallel Computing with CUDA

- CLion clangd bug, YouTrack Issue [CPP-25855](https://youtrack.jetbrains.com/issue/CPP-25855).
  - Incorrect Clangd error for partial template specialization with default parameters (happens within Thrust headers).
    - Ambiguous partial specializations of `pointer_element<pointer<void>>`. 
  - A workaround by @Petr Kudriavtsev:
    - Go to the `Settings | Languages & Frameworks | C/C++ | Clangd`,
    - There will be a field for additional flags which are added to the every compilation command in the project.
    - Add there `-fno-relaxed-template-template-args`.

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

- [C++ Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-compatibility)
  - Host code supports full C++ syntax: 
    - Reference: [3.1.5. C++ Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-compatibility). 
  - Dvice code supports only a subset of C++ syntax: 
    - Listed in [14. C++ Language Support](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-support)

### 📌 [Managing Memory](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

- [Memory Management Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
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
///             - cudaMemcpyDefault
/// Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. 
/// However, cudaMemcpyDefault is only allowed on systems that support unified virtual addressing. 
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
- With `ncu` 
  - Note that `nvprof` is outdated!
  - [Metric Comparasion](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvprof-metric-comparison)



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
  - All threads in a warp *execute* the same instruction at the same time
    - In one clock cycle, one thread
      - either execute the same instruction, 
      - or halt (be idle) for this cycle. 
    - Each thread has its own instruction address counter and register state
    - Each thread carries out the current instruction on its own data
  - Each SM partitions its assigned thread block into warps of 32 threads
    - Warp partition: 
      - Each warp contains threads of consecutive, increasing thread IDs; 
      - The first warp contains thread 0. 
      - Reference: [4.1 SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture)
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
  - `ncu`: Command-line profiler
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
  - These thread blocks are distributed among SMs. 
    - One thread block is scheduled to one SM. 
  - Once a thread block is assigned to a SM
    - Threads in this thread block are further partitioned into warps
    - A warp consists of 32 **consecutive** threads
    - All threads in a warp are executed in SIMT fashion
      - All threads execute the same instruction
      - Each thread carries out that operation on its own private data
- The hardware always allocates a discrete number of warps for a thread block
  - `numberOfWarps = ceil(numberOfThreads / warpSize)`
  - A warp is **never** split between different thread blocks
  - If thread block size is not a multiple of warp size, some threads in the last warp are left inactive
    - But they still consume SM resources, e.g., CUDA cores and registers!

#### 📌 Warp Divergence

- GPU has **no** complex branch prediction mechanism
  - All threads in a warp must execute the same instruction
- Threads in the same warp executing different instructions is referred as *warp divergence*
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

#### 📌 Checking Active Warps

```bash
$ nvcc -O3 --generate-code=arch=compute_75,code=[compute_75,sm_75] sumMatrix.cu -o sumMatrix
$ ncu -k regex:sumMatrixOnGPU2D ./sumMatrix

sumMatrixOnGPU2D <<< (512,512), (32,32) >>> elapsed 60 ms
sumMatrixOnGPU2D <<< (512,1024), (32,16) >>> elapsed 38 ms
sumMatrixOnGPU2D <<< (1024,512), (16,32) >>> elapsed 51 ms
sumMatrixOnGPU2D <<< (1024,1024),(16,16) >>> elapsed 46 ms

sumMatrixOnGPU2D <<<(512,512), (32,32)>>> Achieved Occupancy 0.501071
sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> Achieved Occupancy 0.736900
sumMatrixOnGPU2D <<<(1024,512), (16,32)>>> Achieved Occupancy 0.766037
sumMatrixOnGPU2D <<<(1024,1024),(16,16)>>> Achieved Occupancy 0.810691
```
- Because the second case has more blocks than the first case, 
  it exposed more active warps to the device.
  - This is likely the reason why the second case has a higher achieved occupancy 
    and better performance than the first case.
- The fourth case has the highest achieved occupancy, but it is **not** the fastest. 
  - Therefore, a higher occupancy does **not** always equate to higher performance. 
  - There must be other factors that restrict performance.

#### 📌 Checking Memoty Options

- `C[idx] = A[idx] + B[idx]` has two loads and one store
```bash
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> Global Load Throughput 35.908GB/s
sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> Global Load Throughput 56.478GB/s
sumMatrixOnGPU2D <<<(1024,512), (16,32)>>> Global Load Throughput 85.195GB/s
sumMatrixOnGPU2D <<<(1024,1024),(16,16)>>> Global Load Throughput 94.708GB/s
```
- While the fourth case has the highest load throughput, 
  it is slower than the second case (which only demonstrates around half the load throughput). 
  - A higher load throughput does **not** always equate to higher performance. 
```bash
# The ratio of requested global load throughput to required global load throughput
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> Global Memory Load Efficiency 100.00%
sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> Global Memory Load Efficiency 100.00%
sumMatrixOnGPU2D <<<(1024,512), (16,32)>>> Global Memory Load Efficiency 49.96%
sumMatrixOnGPU2D <<<(1024,1024),(16,16)>>> Global Memory Load Efficiency 49.80%
```
- The load efficiency for the last two cases was half that of the first two cases. 
  - This would explain why the higher load throughput and achieved occupancy 
    of the last two cases did not yield improved performance. 
  - Even though the number of loads being performed is greater for the last two cases, 
    the effectiveness of those loads is lower.
- Note that the common feature for the last two cases is that their block size 
  in the innermost dimension is half of a warp. 
  - **The innermost dimension (`blockDim.x`) should always be a multiple of the warp size**
  - Because otherwise we will have poor global memory access patterns (detailed Chapter 4)

### 🎯 AVOIDING BRANCH DIVERGENCE

- Reduce or avoid branch divergence by rearranging data access patterns
- Practice with a parallel reduction example: E.g., sum an array via airwise parallel sum 
  - A thread sums a pair (containing two elements) and store partial results in-place as input to another iteration
  - Implementations
    - **Neighbored pair**
      - Elements are paired with the immediate neighbor
    - **Interleaved pair** 
      - Paired elements are separated by a given stride
      ```c++
      int recursiveReduce(int * data, int size)
      {
          // terminate check
          if (size == 1)
          {
              return data[0];
          }
          // renew the stride
          const int stride = size / 2;
          // in-place reduction
          for (int i = 0; i != stride; ++i)
          {
              // reduction
              data[i] += data[i + stride];
          }
          // call recursively
          return recursiveReduce(data, stride);
      }
      ``` 

#### 📌 Reduction with Neighbored Pairs

```c++
__global__ void reduceNeighboredLess(int * g_idata, int * g_odata, unsigned int n) 
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int * idata = g_idata + blockIdx.x * blockDim.x;
    // boundary check
    if (n <= idx) 
    {
        return;
    }

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) 
    {
        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x) 
        {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}
```

#### 📌 Reduction with Interleaved Pairs

```c++
// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved(int * g_idata, int * g_odata, unsigned int n) 
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int * idata = g_idata + blockIdx.x * blockDim.x;
    // boundary check
    if (n <= idx) 
    {
        return;
    }

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
    {
        if (tid < stride) 
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}
```

### 🎯 UNROLLING LOOPS

- *Loop unrolling* 
  - A technique that attempts to reduce the frequency of branches and loop maintenance instructions. 
    - Rather than writing the body of a loop once and using a loop to execute it repeatedly, 
      the body is written in code multiple times. 
    - Any enclosing loop then either has its iterations reduced or is removed entirely. 
  - The number of copies made of the loop body is called the *loop unrolling factor*. 
    - The number of iterations in the enclosing loop is divided by the loop unrolling factor. 
```c++
// vanilla
for (int i = 0; i < 100; ++i) 
{
    a[i] = b[i] + c[i];
}

// unrolled
// condition 1 < 100 checked only 50 times (compared to vanilla's 100)
for (int i = 0; i < 100; i += 2) 
{
    // better utilizes spatial locality
    a[i] = b[i] + c[i];
    a[i + 1] = b[i + 1] + c[i + 1];
}
```

#### 📌 Reduction with Unrolling

- Each thread block in the `reduceInterleaved` kernel handles just one portion of the data
  - Manually unrolled the processing of two data blocks by a single thread block
  - For each thread block, data from two data blocks is summed. 
  - This is an example of cyclic partitioning
    - Each thread works on more than one data block and processes a single element from each data block.
```c++
__global__ void reduceUnrolling2(int * g_idata, int * g_odata, unsigned int n) 
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int * idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2 data blocks
    if (idx + blockDim.x < n)
    {
        g_idata[idx] += g_idata[idx + blockDim.x];
        __syncthreads();
    }

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
    {
        if (tid < stride) 
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within thread block
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}
```

#### 📌 Reduction with Unrolled Warps

- `__syncthreads` is used for intra-block synchronization
  - When there are 32 or fewer threads (i.e., a single warp) left. 
  - SIMT means implicit intra-warp synchronization after each instruction.
  - The last 6 iterations of the reduction loop can therefore be unrolled as follows
  ```c++
  if (tid < 32) 
  {
      volatile int * vmem = idata;
      vmem[tid] += vmem[tid + 32];
      vmem[tid] += vmem[tid + 16];
      vmem[tid] += vmem[tid + 8];
      vmem[tid] += vmem[tid + 4];
      vmem[tid] += vmem[tid + 2];
      vmem[tid] += vmem[tid + 1];
  }
  ```
  - This warp unrolling avoids executing loop control and thread synchronization logic.
  - Note that the variable vmem is declared `volatile`
    - Tells the compiler that it must store `vmem[tid]` back to global memory with every assignment. 
    - If `volatile` is omitted, this code will **not** work correctly
      - because the compiler or cache may optimize out some reads or writes to global or shared memory. 
    - If a variable located in global or shared memory is declared `volatile` 
      - the compiler assumes that its value can be changed or used at any time by other threads. 
      - Any reference to `volatile` variables forces a read or write directly to memory 
        - and **not** simply to cache or a register
```c++
__global__ void reduceUnrollWarps8(int * g_idata, int * g_odata, unsigned int n) 
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int * idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) 
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) 
    {
        if (tid < stride) 
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) 
    {
        volatile int * vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}
```

#### 📌 Reduction with Complete Unrolling

- If you know the number of iterations in a loop at compile-time, you can completely unroll it
  - The maximum number of threads per block on either Fermi or Kepler is limited to 1024
  - the loop iteration count in these reduction kernels is based on a thread block dimension
```c++
__global__ void reduceCompleteUnrollWarps8(int * g_idata, int * g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int * idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) 
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) 
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
```

#### 📌 Reduction with Template Functions

- Further reduce branch overhead with templates
  - Compared with `reduceCompleteUnrollWarps8`: Replace block size with template parameter `iBlockSize`
  - The if statements that check the block size will be evaluated at compile time and removed 
    if the condition is not true, resulting in a very effi cient inner loop (like C++'s constexpr if).
```c++
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int * g_idata, int * g_odata, unsigned int n) 
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) 
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32) 
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
```
```c++
switch (blocksize) 
{
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 512:
        reduceCompleteUnroll<512><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 256:
        reduceCompleteUnroll<256><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 128:
        reduceCompleteUnroll<128><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 64:
        reduceCompleteUnroll<64><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
}
```

### 🎯 DYNAMIC PARALLELISM

- So far, all kernels have been invoked from the host thread
- CUDA *Dynamic Parallelism* allows new GPU kernels to be created and synchronized directly on the GPU
  - Make recursive algorithms more transparent and easier to understand
  - Postpone the decision of exactly how many blocks and grids to create on a GPU until runtime
    - Taking advantage of the GPU hardware schedulers and load balancers
      dynamically and adapting in response to data-driven decisions or workloads
  - Reduce the need to transfer execution control and data between the host and device

#### 📌 Nested Execution

- The same kernel invocation syntax is used to launch a new kernel within a kernel
- Kernel executions are classified into two types
  - Parent
    - A *parent thread*, *parent thread block*, or *parent grid* has launched a new grid, the child grid. 
    - A parent is **not** considered complete until all of its child grids have completed.
  - Child
    - A *child thread*, *child thread block*, or *child grid* has been launched by a parent. 
    - A child grid must complete before the parent thread, parent thread block, or parent grids are considered complete. 
- Runtime guarantees an implicit synchronization between the parent and the child. 
- Grid launches in a device thread are visible across a thread block. 
  - A thread may synchronize on the child grids launched by that thread or by other threads in the same thread block.
  - Execution of a thread block is **not** considered complete until 
    all child grids created by all threads in the block have completed. 
  - If all threads in a block exit before all child grids have completed, 
    implicit synchronization on those child grids is triggered.
- When a parent launches a child grid, the child is **not** guaranteed to begin execution 
  until the parent thread block explicitly synchronizes on the child.
- Memory
  - Parent and child grids share the same global and constant memory storage
  - Parent and child grids have distinct local and shared memory
    - Shared and local memory are private to a thread block or thread
      - **not** visible or coherent between parent and child. 
    - Local memory is private storage for a thread
      - **not** visible outside of that thread. 
    - It is **invalid** to ~~pass a pointer to local memory as an argument when launching a child grid~~.
  - Parent and child grids have concurrent access to global memory
    - with weak consistency guarantees between child and parent. 
      - There are two points in the execution of a child grid 
        when its view of memory is fully consistent with the parent thread: 
        - at the start of a child grid,
        - when the child grid completes.
    - All global memory operations in the parent thread prior to a child grid invocation
      are guaranteed to be visible to the child grid.
    - All memory operations of the child grid are guaranteed to be visible to the parent 
      after the parent has synchronized on the child grid’s completion.

#### 📌 Nested Hello World on the GPU

```c++
__global__ void nestedHelloWorld(int const iSize, int iDepth) 
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid, blockIdx.x);
    // condition to stop recursive execution
    if (iSize == 1) return;
    // reduce block size to half
    int nthreads = iSize >> 1;
    // thread 0 launches child grid recursively
    if (tid == 0 && nthreads > 0) 
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}
```
- Restrictions on Dynamic Parallelism: 
  - Only supported by devices of compute capability 3.5+. 
  - Kernels invoked through dynamic parallelism can **not** be launched on physically separate devices. 
    - However, it is permitted to query properties for any CUDA capable device in the system.
  - The maximum nesting depth of dynamic parallelism is limited to 24. 
    - In reality, most kernels will be limited by the amount of memory
      required by the device runtime system at each new level, 
      as the device runtime reserves additional memory for synchronization management
      between every parent and child grid at each nested level.

#### 📌 Nested (Recursive) Reduction

- Vanilla version
  - 2,048 blocks initially 
  - Each block performs 8 recursions, 16,384 child blocks were created
    - intra-block synchronization with `__syncthreads` was also invoked 16,384 times 
  - poor kernel performance
```c++
__global__ void gpuRecursiveReduce(int * g_idata, int * g_odata, unsigned int isize) 
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int * idata = g_idata + blockIdx.x * blockDim.x;
    int * odata = &g_odata[blockIdx.x];

    // stop condition
    if (isize == 2 && tid == 0) 
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // nested invocation
    int istride = isize >> 1;

    if (istride > 1 && tid < istride) 
    {
        // in place reduction
        idata[tid] += idata[tid + istride];
    }
    // sync at block level
    __syncthreads();

    // nested invocation to generate child grids
    if (tid == 0) 
    {
        gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
        // sync all child grids launched in this block
        cudaDeviceSynchronize();
    }
    // sync at block level again
    __syncthreads();
}
```
- No Sync Version
  - When a child grid is invoked, its view of memory is fully consistent with the parent thread. 
  - Each child thread only needs its parent’s values to conduct the partial reduction. 
  - the in-block synchronization performed before the child grids are launched is unnecessary. 
  - Still poor performance
    - Each block generates a child grid, resulting in a huge number of invocations. 
```c++
__global__ void gpuRecursiveReduceNosync(int * g_idata, int * g_odata, unsigned int isize) 
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int * idata = g_idata + blockIdx.x * blockDim.x;
    int * odata = &g_odata[blockIdx.x];

    // stop condition
    if (isize == 2 && tid == 0) 
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // nested invoke
    int istride = isize >> 1;

    if (istride > 1 && tid < istride) 
    {
        idata[tid] += idata[tid + istride];

        if (tid == 0) 
        {
            gpuRecursiveReduceNosync<<<1, istride>>>(idata, odata, istride);
        }
    }
}
```
- Version 2
  - The first thread in the first block of a grid invokes the child grids for each nested step
```c++
__global__ void gpuRecursiveReduce2(int * g_idata, int * g_odata, int iStride, int const iDim) 
{
    // convert global data pointer to the local pointer of this block
    int * idata = g_idata + blockIdx.x * iDim;
    // stop condition
    if (iStride == 1 && threadIdx.x == 0) 
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // in place reduction
    idata[threadIdx.x] += idata[threadIdx.x + iStride];
    // nested invocation to generate child grids
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
        gpuRecursiveReduce2<<<gridDim.x, iStride / 2>>>(g_idata, g_odata, iStride / 2, iDim);
    }
}
```



## 🌱 4 Global Memory

### 🎯 INTRODUCTION

#### 📌 Memory Hierachy

- *Principle of Locality*
  - Temporal locality
    - If a data location is referenced, 
      then it is more likely to be referenced again within a short time period
      and less likely to be referenced as more and more time passes
  - Spatial locality
    - If a memory location is referenced, 
      nearby locations are likely to be referenced as well
- *Memory Hierachy*: Speed fast to low, size small to large
  - Registers
  - Caches
    - Lower-latency memory, such as CPU L1 cache
    - Implemented using *SRAM* (Static Random Access Memory)
  - Main Memory
    - Both CPUs and GPUs use *DRAM* (Dynamic Random Access Memory)
  - Disk Memory
    - Implemented using magnetic disks, flash drives, etc. 
    - Properties
      - Lower cost per bit
      - Higher capacity
      - Higher latency
      - Less frequently accessed by the processer

#### 📌 CUDA Memory Model

- Two types of memory
  - Programmable memory
    - You explicitly control what data is placed in programmable memory
  - Non-programmable memory
    - You have **no** control over data placement
    - You rely on automatic techniques to achieve good performance
    - E.g., CPU's L1/L2 cache
- Memory Layout on an SM ([Hardware Model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-hw-model))
  - The SM is designed to simultaneously execute multiple thread blocks. 
    - Thread blocks can be from different grid launches.
  - Each SM is partitioned into four processing blocks, called SM *sub partitions*. 
    - The primary processing elements. 
    - Each sub partition contains the following units:
      - Warp scheduler
      - Register File
      - Execution Units, Pipelines, and Cores
        - Integer Execution Units
        - Floating Point Execution Units
        - Memory Load/Store Units
        - Special Function Unit
        - Tensor Cores
  - Shared within an SM across the four SM partitions are:
    - Unified L1 Data Cache / Shared Memory
    - Texture Units
    - RT Cores
- Warp
  - All warps from the same thread block will also be assigned on the same SM. 
  - Allocated to a sub partition and resides on the sub partition from launch to completion. 
  - A warp is referred to as active or resident when it is mapped to a sub partition. 
  - A sub partition manages a fixed size pool of warps. 
    - 16 for the Volta architecture;
    - 8 for the Turing architecture. 
    - E.g., NVIDIA Geforce RTX 2080 Ti (Turing architecture)
      - 68 Multiprocessors, 64 CUDA Cores/MP, In total 4352 CUDA Cores
      - Each SM has 4 sub partitions
        - Each sub partition has 16 CUDA Cores, and manages 8 warps
      - Each SM could host 32 active warps (1024 threads)
        - Maximum number of threads per multiprocessor:  1024
        - Maximum number of threads per block:           1024
  - Active warps can be in eligible state if the warp is ready to issue an instruction. 
    - The warp has a decoded instruction; 
    - All input dependencies resolved; 
    - The function unit is available. 
  - A warp is stalled when the warp is waiting on
    - An instruction fetch,
    - A memory dependency (result of memory instruction),
    - An execution dependency (result of previous instruction), or
    - A synchronization barrier.
- Programmable memories in the CUDA memory model
  - Registers
    - On-chip
    - Private to a thread in a kernel
    - Lifetime: same as the kernel
  - Local memory
    - Device memory
    - Private to a thread in a kernel
    - Lifetime: same as the kernel
  - Shared memory
    - On chip, for each SM, share a 64KB memory-block with L1 cache
    - Private to a thread block
    - Visible to all threads in the same thread block
    - Lifetime: same as the thread block
  - Constant memory
    - On-chip, one part of the cache memory
    - Read-only
    - Accessible by all threads and the host
    - Lifetime: same as the application
  - Texture memory
    - On-chip, one part of the cache memory
    - Read-only
    - Accessible by all threads and the host
    - Lifetime: same as the application
  - Global memory
    - Device memory
    - Accessible by all threads and the host
    - Lifetime: same as the application
- **Registers**
  - Fastest
  - Automatic variables declared in a kernel **without** any other type qualifiers are generally stored in registers
  - Arrays declared in a kernel may also be stored in registers
    - Only if the indices used to reference the array are compile-time constants
  - Check hardware resources used by a kernel by `nvcc` options `-Xptxas -v,-abi=no`
  - If a kernel uses more registers than the hardware limit, the excess registers will spill over to local memory 
    - Can have adverse performance consequences. 
    - `nvcc` uses heuristics to minimize register usage and avoid register spilling
    - You can optionally aid these heuristics in the form of *launch bounds*
      - `maxThreadsPerBlock`: The maximum number of threads per block that a kernel will launch.
      - `minBlocksPerMultiprocessor`: optional, the desired minimum number of resident blocks per SM. 
      - Optimal launch bounds for a given kernel will usually differ across major architectural revisions.
    ```c++
    __global__ void
    __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiProcessor)
    kernel(...) 
    {
        // Your kernel body
    }
    ```
    - You can also control the maximum number of registers used by all kernels in a compilation unit
      using the `maxrregcount` compiler option, e.g., `-maxregcount=32`
      - `maxregcount` is ignored for any kernels with launch bounds. 
- **Local Memory**
  - Variables in a kernel that are eligible for registers but spilled into local memory, e.g.,
      - Local arrays references with non-compiler-constant indices
      - Large local structures or arrays that would consume too much register space
      - Any variable that does not fit within the kernel register limit
  - *local* is misleading 
    - Values spilled to local memory reside in the same physical location as global memory
    - Local memory accesses are characterized by high latency and low bandwidth 
    - Local memory accesses are subject to the requirements for efficient memory access
      - To be detailed in the section Memory Access Patterns found later in this chapter. 
    - Local memory data is also cached in a per-SM L1 and per-device L2 cache.
- *Shared Memory*
  - Variables decorated with `__shared__` attribute in a kernel
  - Shared memory is on-chip
    - Higher bandwidth and lower latency (compared with local/global memory)
    - Used similarly to CPU L1 cache (but programmable)
  - Each SM has a limited amount of shared memory partitioned among thread blocks
    - **Don't** overuse or will limit the number of active warps
  - Lifetime: Same as thread block
    - Even though declared in the scope of a kernel function
  - Serves as a basic means for inter-thread communication
    - Threads within a block can cooperate by sharing data stored in shared memory
    - Access to shared memory must be synchronized using `__syncthread()`;
  - L1 cache and shared memory for an SM use the same 64KB of on-chip memory
    - Statically partitioned
    - Can be dynamically configured at runtime
    ```c++
    /// Configures the partitioning of on-chip memory for kernel function func
    /// @param func        The kernel function to configure
    /// @param cacheConfig Takes one of the following values
    ///                    - cudaFuncCachePreferNone:   no preference (default)
    ///                    - cudaFuncCachePreferShared: prefer 48KB shared memory and 16KB L1 cache
    ///                    - cudaFuncCachePreferL1:     prefer 48KB L1 cache and 16KB shared memory
    ///                    - cuadFuncCachePreferEqual:  prefer 32KB L1 cache and 32KB shared memory
    cudaError_t cudaFuncSetCacheConfig(const void * func, enum cudaFuncCache cacheConfig);
    ```
- **Constant Memory**
  - Variables decorated with `__constant__` attribute in global scope
    - Must in global scope, outside of any kernels
    - Statically declared and visible to all kernels in the same compilation unit
    - Read-only to kernels
      - Hence must be initialized by the host using [cudaMemcpyToSymbol](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9bcf02b53644eee2bef9983d807084c7)
      ```c++
      /// Copies count bytes from the memory pointed to by src to the memory pointed to by symbol. 
      /// This function is synchronous in most cases. 
      /// @param symbol A variable that resides on the device in global or constant memory. 
      /// @param src    ...
      /// @param count  ...
      __host__ cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count);
      ```
      - A direct call to `cudaMemcpy` is illegal without prior symbol lookup
  - Resides in device memory and cached in per-SM constant cache
    - Limited amount: 64KB for all compute capabilities
  - Performs best when all threads in a warp read from the same memory address
    - E.g., Coefficient for a math formula
    - If each thread in a warp reads from different addresses and only reads once, 
      then constant memory is **not** a good choice
      - A single read from constant memory broadcasts to all threads in a warp
- **Texture Memory**
  - Resides in device memory and cached in per-SM read-only cache
    - Global memory
    - Accessed through a dedicated read-only cache. 
      - includes support for hardware filtering
      - can perform floating-point interpolation as part of the read process
  - Optimized for 2D spatial locality
    - Threads in a warp that use texture memory to access 2D data will achieve the best performance. 
    - For some applications, provides performance advantages due to the cache and the filtering hardware.
    - For other applications using texture memory can be slower than global memory.
- **Global Memory**
  - Global w.r.t. scope and lifetime 
    - Its state can be accessed on the device from any SM throughout the lifetime of the application.
  - Can be declared either statically or dynamically
    - Static: in device code with qualifier `__device__`
    - Dynamic: by host using `cudaMalloc` and `cudaFree`
  - Take care when accessing global memory from multiple threads. 
    - Thread execution can **not** be synchronized across thread blocks
    - Potential hazard of data race (*undefined behavior*)
  - Resides in device memory
    - Accessible via 32-Byte, 64-Byte, or 128-Byte memory transactions
    - These transactions must be naturally aligned
      - The first address must be a multiple of 32 Bytes, 64 Bytes, or 128 Bytes
  - Number of transactions required to satisfy a warp memory request (load/store operation)
    - Depends on two factors
      - Distribution of memory addresses across the threads of that warp
      - Alignment of memory addresses per transaction
- **GPU Caches**
  - Non-programmable memory (like CPU caches)
  - Four types of cache
    - L1
      - One per SM
      - Stores data in local and global memory (including register spills)
    - L2
      - One shared by all SMs
      - Stores data in local and global memory (including register spills)
    - Read-only constant
      - One per SM
    - Read-only texture
      - One per SM
  - Only memory load operations can be cached
    - On CPU, both store and load can be cached
- **Static Global Memory**
  - Sample
  ```c++
  #include <iostream>
  #include <cuda_runtime.h>

  __device__ float devData;

  __global__ void checkGlobalVariable()
  {
      // display the original value
      printf("Device: the value of the global variable is %f\n", devData);
      // alter the value
      devData += 2.0f;
  }

  int main()
  {
      // initialize the global variable
      float value = 3.14f;
      cudaMemcpyToSymbol(devData, &value, sizeof(float));
      printf("Host: copied %f to the global variable\n", value);
      // invoke the kernel
      checkGlobalVariable<<<1, 1>>>();
      // copy the global variable back to the host
      cudaMemcpyFromSymbol(&value, devData, sizeof(float));
      printf("Host: the value changed by the kernel to %f\n", value);
      cudaDeviceReset();
      return EXIT_SUCCESS;
  }
  ```
  - Even though the host and device code are stored in the same file, 
    they exist in completely different worlds. 
  - The host code can **not** directly access a device variable 
    even if it is visible in the same file scope. 
  - Similarly, device code can **not** directly access a host variable either.
  - Notes
    - `cudaMemcpyToSymbol` is in the CUDA runtime API 
      and uses GPU hardware behind the scenes to perform the access.
      - The variable `devData` is passed here as a *symbol*, 
        **not** as the address of the variable in device global memory.
      - In the kernel, `devData` is used as a variable in global memory. 
    - `cudaMemcpy` can **not** be used to transfer data into `devData` using the address of the variable:
    ```c++
    cudaMemcpy(&devData, &value, sizeof(float),cudaMemcpyHostToDevice);
    ```
    - You can **not** use the reference operator `&` on a device variable from the host, 
      because it is simply a symbol representing the physical location on the GPU. 
      A direct call to `cudaMemcpy` is **illegal** without prior symbol lookup.
      - However, you can acquire the address of a global variable
        by explicitly making a call using the following CUDA API:
      ```c++
      /// Fetches the physical address of the global memory associated with the provided device symbol
      cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol);
      ```
      ```c++
      float * dptr = nullptr;
      cudaGetSymbolAddress(reintrepret_cast<void **>(&dptr), devData);
      cudaMemcpy(dptr, &value, sizeof(float), cudaMemcpyHostToDevice);
      ```
    - CUDA pinned memory
      - A single exception to being able to directly reference GPU memory from the host
      - Both host code and device code can access pinned memory directly by simply dereferencing a pointer. 
      - Detailed in the next section

### 🎯 MEMORY MANAGEMENT

#### 📌 Memory Allocation and Deallocation

- [Memory Management Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
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
///             - cudaMemcpyDefault
/// Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. 
/// However, cudaMemcpyDefault is only allowed on systems that support unified virtual addressing. 
__host__ cudaError_t cudaMemcpy(void * dst, void * src, size_t count, cudaMemcpyKind kind);

/// Initializes or sets device memory to a value.
__host__ cudaError_t cudaMemset(void * devPtr, int value, size_t count);

/// Frees memory on the device.
__host__ ​__device__ cudaError_t cudaFree(void * devPtr);

/// Convert an error code to human-readable error message. 
/// Returns the description string for an error code.
__host__ ​__device__ ​const char * cudaGetErrorString(cudaError_t error);
```
- Device memory allocation and deallocation are expensive operations.  
- Device memory should be reused for performance.

#### 📌 Memory Transfer

- Data transfer between host and device is done by [cudaMemcpy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8). 
- Can throttle overall application. 
- Minimize host-device transfers!

#### 📌 Pinned Memory

- Allocated host memory is by default *pageable*. 
  - Subject to page fault operations by host OS. 
- GPU can **not** access data in pageable host memory. 
  - GPU has **no** control over OS page fault mechnism. 
- *Page-locked memory* or *Pinned* memory (on host). 
  - Allocated by CUDA driver temporarily when transferring pageable data from host to device. 
  - First, copy source host data to pinned memory;
  - Then, transfer from pinned memory to device memory. 
- [cudaMallocHost](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gab84100ae1fa1b12eaca660207ef585b)
  ```c++
  /// Allocates page-locked (pinned) host memory that is accessible to device.
  /// @param ptr  - Pointer to allocated host memory. 
  /// @param size - Requested allocation size in bytes. 
  __host__ ​cudaError_t cudaMallocHost(void ** ptr, size_t size);
  ```
  - Pinned memory must be freed with [cudaFreeHost](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g71c078689c17627566b2a91989184969). 
  ```c++
  if (cudaMallocHost(reinterpret_cast<void **>(&h_aPinned), bytes) != cudaSuccess)
  {
      fprintf(stderr, "Error returned from pinned host memory allocation\n");
      exit(1);
  }
  // do something...
  cudaFreeHost(h_aPinned);
  ```
- Pinned memory can be accessed directly from device
  - Higher bandwidth for read/write operations than pageable memory
  - More expensive to allocate and deallocate than pageable memory
  - Using too much pinned memory would occupy the amount of pageable memory available to the host 
    - Hinders host from sotring virtual memory data
    - Downgrades performance

#### 📌 Zero-Copy Memory

- *Zero-copy memory*
  - Both host and device can directly access zero-copy memory
    - Pinned (non-pageable, page-locked) memory that is mapped into the device address space
  - GPU threads can directly access zero-copy memory
    - Leveraging host memory when there is insufficient device memory
    - Avoiding explicit data transfer between the host and device
    - Improving PCIe transfer rates
  - Must synchronize memory accesses across the host and the deive
    - Potential data hazards caused by multiple threads accessing the same memory location without synchronization. 
    - Undefined behavior when data race happens
  - Create zero-copy memory via [cudaHostAlloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902)
  ```c++
  /// Allocates count bytes of host memory that is page-locked and accessible to the device.
  /// Memory allocated by this function must be freed with cudaFreeHost. 
  /// The flags parameter enables further configuration of special properties of the allocated memory. 
  /// @param pHost - Device pointer to allocated memory; 
  /// @param count - Requested allocation size in bytes; 
  /// @param flags - Requested properties of allocated memory. 
  ///                Takes one of the following four values:
  ///                - cudaHostAllocDefault:       Makes behavior of cudaHostAlloc identical to cudaMallocHost;
  ///                - cudaHostAllocPortable:      Returns pinned memory that can be used by all CUDA contexts, 
  ///                                              not just the one that performed the action;
  ///                - cudaHostAllocMapped:        Maps the allocation into the CUDA address space. 
  ///                                              The device pointer to the memory may be obtained 
  ///                                              by calling cudaHostGetDevicePointer;
  ///                - cudaHostAllocWriteCombined: Returns write-combined memory, 
  ///                                              which can be transferred across the PICe bus 
  ///                                              more quickly on some systems 
  ///                                              but can not be read efficiently by most hosts. 
  ///                                              A good choice for buffers that will be 
  ///                                              written by the host and read by the device 
  ///                                              using mapped pinned memory or host-to-device transfers;
  __host__ cudaError_t cudaHostAlloc(void ** pHost, size_t count, unsigned int flags);
  ```
  - Obtain device pointer for the mapped pinned memory via
  ```c++
  /// Returns a device pointer in pDevice 
  /// that can be referenced on the device to access mapped, pinned host memory. 
  /// This function will fail if the device does not support mapped, pinned memory. 
  /// @param flag Reserved for future use, must be set to zero.
  cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags);
  ```
  - Using zero-copy memory as a supplement to device memory with frequent read/write operations 
    will significantly **slow down** performance
    - Every memory transaction to mapped memory must pass over the PCIe bus, 
    - A significant amount of latency is added even when compared to global memory.
  - A good choice for sharing a small amount of data between the host and device
    - Simplifies programming and offers reasonable performance. 
  - For larger datasets with discrete GPUs connected via the PCIe bus, 
    zero-copy memory is a **poor** choice and causes significant performance degradation.
  - P.S. 
    - Two common categories of heterogeneous computing system architectures:
      - Integrated
        - CPUs and GPUs are fused onto a single die and physically share main memory. 
        - Zero-copy memory is more likely to benefit both performance and programmability 
          - because no copies over the PCIe bus are necessary.
      - Discrete
        - Devices connected to the host via PCIe bus
        - Zero-copy memory is advantageous only in special cases. 

#### 📌 Unified Virtual Addressing

- *Unified Virtual Addressing* (UVA)
  - Supported on 64-bit Linux systems
  - Host memory and device memory share a single virtual address space
    - The memory space referenced by a pointer becomes transparent to application code!
  - Prior to UVA: Multiple memory spaces
    - You need to manage which pointers refereed to host memory and which referred to device memory
- Under UVA
  - Pinned host memory allocated with `cudaHostAlloc` has identical host and device pointers
  - You can pass the returned pointer directly to a kernel function 
    - **without** ~~the need of calling `cudaHostGetDevicePointer` to get another device pointer~~

#### 📌 Unified Memory

- *Unified Memory*
  - Creates a pool of managed memory,
    where each allocation from this memory pool is accessible on both the CPU and GPU 
    with the same memory address (pointer)
  - The underlying system automatically migrates data in the unified memory space between the host and device
    - This data movement is transparent to the application, greatly simplifying the application code
  - Depends on UVA support
    - UVA provides a single virtual memory address space for all processors in the system. 
    - UVA does **not** automatically migrate data from one physical location to another
      - unique to Unified Memory.
- Unified Memory offers a *single-pointer-to-data* model 
  - Unified Memory is conceptually similar to zero-copy memory. 
    - Zero-copy memory is allocated in host memory
    - Kernel performance generally suffers from high-latency accesses to zero-copy memory over the PCIe bus.
  - Unified Memory decouples memory and execution spaces 
    - so that data can be transparently migrated on demand to the host or device to improve locality and performance.
- *Managed Memory*
  - Unified Memory allocations that are automatically managed by the underlying system
  - Is interoperable with device-specific allocations such as those created using the `cudaMalloc`.
  - You can use both types of memory in a kernel: 
    - managed memory that is controlled by the system
    - un-managed memory that must be explicitly allocated and transferred by the application. 
  - All CUDA operations that are valid on device memory are also valid on managed memory. 
    - The primary difference is that the host is also able to reference and access managed memory.
- Managed memory can be allocated statically or dynamically. 
  - Statically: As device variable in global scope
    - Add a `__managed__` annotation to device variable declaration. 
    - This can only be done in file-scope and global-scope. 
    - The variable can be referenced directly from either host or device code
    ```c++
    __device__ __managed__ int y;
    ```
  - Dynamically: In host code
    - Device code can **not** call [cudaMallocManaged](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b)
  ```c++
  /// Allocates memory that will be automatically managed by the Unified Memory system,  
  /// and returns a pointer which is valid on all devices and the host. 
  /// The behavior of a program with managed memory is functionally unchanged 
  /// to its counterpart with un-managed memory. 
  /// However, a program that uses managed memory can take advantage of 
  /// automatic data migration and duplicate pointer elimination.
  /// @param devPtr Pointer to allocated device memory;
  /// @param size   Requested allocation size in bytes;
  /// @param flags  Must be either cudaMemAttachGlobal (default) or cudaMemAttachHost. 
  ///               - cudaMemAttachGlobal: This memory is accessible from any stream on any device.
  __host__ cudaError_t cudaMallocManaged(void ** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
  ```

### 🎯 MEMORY ACCESS PATTERNS

#### 📌 Aligned and Coalesced Access

- Memory Hierachy
  - SM (chip includes the following)
    - Registers (on chip)
    - On-chip Cache
      - L1 Cache and Shared Memory (SMEM) (share 64KB cache storage)
      - Constant Memory
      - Read-only Memory
  - L2 Cache (off chip)
  - DRAM (off-chip device memory)
    - Local Memory
    - Global Memory
- Global memory 
  - a logical memory space that you can access from your kernel.
  - loads/stores are staged through caches. 
  - Requests from a warp are packed and serviced together. 
  - Kernel memory requests: Either 128-byte or 32-byte memory transactions.
    - 128-byte: Both L1 and L2 cache is used
    - 32-byte: Only L2 cache is used
- Two characteristics of device memory accesses for performance: 
  - Aligned memory accesses
    - Occur when the first address of a device memory transaction 
      is an even multiple of the cache granularity for the transaction
      (either 32 bytes for L2 cache or 128 bytes for L1 cache). 
      Performing a misaligned load will cause wasted bandwidth.
  - Coalesced memory accesses
    - Occur when all 32 threads in a warp access a contiguous chunk of memory.
  - To maximize global memory throughput, organize memory operations to be both aligned and coalesced. 

#### 📌 Global Memory Reads

- Three cache paths to pipepile global memory onto chip:
  - L1/L2 cache: Default path. 
    - L1 cache is enabled by default, could be manually enabled/disabled by `nvcc` flags (flag details later). 
    - Only L2: Query, if miss then 32-byte memory transactions. 
    - Both L1 and L2: Query L1, if miss then query L2, if still miss then 128-byte memory transactions.
  - Constant cache
  - Read-only cache
- **Memory Load Access Patterns**
  - Two types of memory loads:
    - Cached load (L1 cache enabled)
    - Uncached load (L1 cache disabled)
  - Access patterns for memory loads:
    - Cached vs uncached: Cached if L1 cache is enabled. 
    - Aligned vs misaligned: Aligned if the first address of a memory access is a multiple of 32 bytes. 
    - Coalesced vs uncoalsced: Coalesced if a warp accesses a contiguous chuck of data. 
- Cached Loads
  - Pass through L1 cache
  - Device memory transaction granularity: L1 cache line (128 bytes)
  - All memory accesses from a warp is collected together, then serviced. 
  - These memory accesses does not consecutive by thread ID. 
  - **GPU L1 Cache vs CPU L1 Cache**
    - CPU L1 Cache: Optimized for both spatial and temporal locality. 
    - GPU L1 Cache: Only for spatial locality. 
      - Frequent access to a cached L1 memory location does **not** increase its chance to stay in cache. 
  - Explicitly enable L1 cache: `nvcc` flags `-Xptxas -dlcm=ca`
- Uncached Loads
  - Does **not** ass through L1 cache
  - Granualarity: Memory segments (32 bytes)
    - More fine-grained loads. 
    - Better bus utilization for misaligned or uncoalesced memory accesses. 
  - Force uncached loads: `nvcc` flags `-Xptxas -dlcm=cg`
- Efficiency
  - $\mathrm{gld_efficiency} = \dfrac{\mathrm{RequestedGlobalMemoryLoadThroughput}}{\mathrm{RequiredGlobalMemoryLoadThroughput}}$
- Read-only Cache
  - Was originally reserved for use by texture memory loads (prior to compute capability 3.5). 
  - Granularity: 32 bytes.
  - Two ways to direct memory reads through the read-only cache:
    - Use the function `__ldg`;
    - Use the declaration qualifier `const __restrict__` on the pointer being dereferenced. 
  ```c++
  __global__ void copyKernel(int * out, int * in) 
  {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      out[idx] = in[idx];
  }

  /// Use the intrinsic function __ldg to direct 
  /// the read accesses for array in through the readonly cache. 
  __global__ void copyKernelVersion2(int * out, int * in) 
  {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      out[idx] = __ldg(&in[idx]);
  }

  /// Apply const __restrict__ qualifiers to pointers. 
  /// These qualifiers help the nvcc compiler recognize non-aliased pointers 
  /// (that is, pointers which are used exclusively to access a particular array). 
  /// nvcc will automatically direct loads from non-aliased pointers through the read-only cache.
  __global__ void copyKernelVersion3(int * __restrict__ out, const int * __restrict__ in) 
  {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      out[idx] = in[idx];
  }
  ```

#### 📌 Global Memory Writes

- Memory store operations: 
  - Use only L2 cache;
  - Write-allocate policy;
  - 32-byte segment granularity, 1/2/3/4 segments at a time.
  - E.g., two addresses fall within the same 128-byte region but not within an aligned 64-byte region: 
    - One four-segment transaction will be issued. 
    - I.e., a single four-segment transaction performs better than two one-segment transactions. 

#### 📌 Arrays of Structures vs Structures of Arrays

- SIMD-style parallel programming paradigms prefer SoA. 
- In CUDA C programming, SoA is also typically preferred. 
  - Data elements are pre-arranged for efficient coalesced access to global memory. 
- *Array of structure* (AoS)
```c++
struct InnerStruct
{
    float x;
    float y;
};

__global__ void testInnerStruct(InnerStruct * data, InnerStruct * result, const int n) 
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) 
    {
        InnerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

/// gld_efficiency 50.00%
/// gst_efficiency 50.00%
/// Both load and store memory requests are replayed for the AoS data layout. 
/// The fields x and y are stored adjacent in memory and have the same size.  
/// Every time a memory transaction is performed to load the values of a particular field, 
/// exactly half of the bytes loaded must also belong to the other field. 
/// Thus, 50 percent of all required load and store bandwidth is unused. 
```
- *Structure of Array* (SoA)
```c++
struct InnerArray
{
    float x[kN];
    float y[kN];
};

__global__ void testInnerArray(InnerArray * data, InnerArray * result, const int n) 
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) 
    {
        float tmpx = data->x[i];
        float tmpy = data->y[i];
        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}

/// gld_efficiency 100.00%
/// gst_efficiency 100.00%
```
- [Local Test Result](./examples/pccp/pccp_171_aos_vs_soa.cu)

#### 📌 Performance Tuning

- Two goals when optimizing memory bandwidth:
  - Aligned and coalesced memory accesses
    - Reduce wasted bandwidth
  - Sufficient concurrent memory operations
    - Hide memory latency
    - Increasing the number of independent memory operations performed within each thread.
    - Experimenting with the execution configuration of a kernel launch to expose sufficient parallelism to each SM.
- Unrolling Techniques
  - Unrolling loops that contain memory operations adds more independent memory operations to the pipeline. 
    - For an I/O-bound kernel, exposing sufficient memory access parallelism is a high priority. 
  - Consider the earlier `readSegment` example. 
    - Revise the `readOffset` kernel such that each thread performs four independent memory operations. 
    - Because each of these loads is independent, you can expect more concurrent memory accesses.
    - This unrolling technique has a tremendous impact on performance, even **more than address alignment**.
  ```c++
  __global__ void readOffset(float * A, float * B, float * C, const int n, int offset) 
  {
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int k = i + offset;

      if (k < n) 
      {
          C[i] = A[k] + B[k];
      }
  }

  __global__ void readOffsetUnroll4(float * A, float * B, float * C, const int n, int offset) 
  {
      unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
      unsigned int k = i + offset;

      if (k + 3 * blockDim.x < n) 
      {
          C[i] = A[k]
          C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
          C[i + 2 * blockDim.x] = A[k + 2 * blockDim.x] + B[k + 2 * blockDim.x];
          C[i + 3 * blockDim.x] = A[k + 3 * blockDim.x] + B[k + 3 * blockDim.x];
      }
  }
  ```
  ```
  // $ ./readSegment 0
  // readOffset <<< 32768, 512 >>> offset 0 elapsed 0.001820 sec
  // $ ./readSegment 11
  // readOffset <<< 32768, 512 >>> offset 11 elapsed 0.001949 sec
  // $ ./readSegment 128
  // readOffset <<< 32768, 512 >>> offset 128 elapsed 0.001821 sec

  // $ ./readSegmentUnroll 0
  // unroll4 <<< 8192, 512 >>> offset 0 elapsed 0.000599 sec
  // $ ./readSegmentUnroll 11
  // unroll4 <<< 8192, 512 >>> offset 11 elapsed 0.000615 sec
  // $ ./readSegmentUnroll 128
  // unroll4 <<< 8192, 512 >>> offset 128 elapsed 0.000598 sec

  // readOffset gld_efficiency 49.69%
  // readOffset gst_efficiency 100.00%
  // readOffsetUnroll4 gld_efficiency 50.79%
  // readOffsetUnroll4 gst_efficiency 100.00%

  // readOffset gld_transactions 132384
  // readOffset gst_transactions 32928
  // readOffsetUnroll4 gld_transactions 33152
  // readOffsetUnroll4 gst_transactions 8064
  ```
- Exposing More Parallelism
  - Experiment with the grid and block size of a kernel. 
```
$ ./readSegmentUnroll 11 1024 22
unroll4 <<< 1024, 1024 >>> offset 11 elapsed 0.000184 sec
$ ./readSegmentUnroll 11 512 22
unroll4 <<< 2048, 512 >>> offset 11 elapsed 0.000162 sec
$ ./readSegmentUnroll 11 256 22
unroll4 <<< 4096, 256 >>> offset 11 elapsed 0.000162 sec
$ ./readSegmentUnroll 11 128 22
unroll4 <<< 8192, 128 >>> offset 11 elapsed 0.000162 sec
```
- **MAXIMIZING BANDWIDTH UTILIZATION**
  - Two major factors on the performance of device memory operations:
    - Efficient use of bytes moving between device DRAM and SM on-chip memory:
      - Memory access patterns should be aligned and coalesced.
    - Number of memory operations concurrently in-flight: 
      - Unrolling, yielding more independent memory accesses per thread; 
      - Modifying the execution confiuration of a kernel launch.

### 🎯 WHAT BANDWIDTH CAN A KERNEL ACHIEVE?

- *Memory Latency*
  - The time to satisfy an individual memory request. 
  - Hiding memory latency by maximizing the number of concurrently executing warps. 
- *Memory Bandwidth*
  - The rate at which device memory can be accessed by an SM. 
  - Maximizing memory bandwidth efficiency by properly aligning and coalescing memory accesses.
- Sometimes a bad access pattern is inherent to the nature of the problem at hand. 
- Options for an inherently imperfect access pattern. 

#### 📌 Memory Bandwidth

- Most kernels are are memory bandwidth-bound.
- *Theoretical Bandwidth*
  - The absolute maximum bandwidth achievable for the hardware. 
- *Effective Bandwidth*
  - The measured bandwidth that a kernel actually achieves. 
  - $\mathrm{Effective Bandwidth (GB/s)} = \dfrac{\mathrm{Bytes Read (Byte)} + \mathrm{Bytes Written (Byte)}}{10^{9} \mathrm{(Byte/GB)} \times \mathrm{Times Elapsed (s)}}$
  - E.g., if copying a $2048 \times 2048$ floating point matrix, $\mathrm{Effective Bandwidth} = \dfrac{2048 \times 2048 \times \mathrm{sizeof}(\mathbf{float}) \times 2}{10^9 \times \mathrm{TimesElapsed(s)}}$
- Matrix Transpose Problem
  - Sample host implementation
  ```c++
  /// x: Row; y: Column
  void transposeHost(float * out, const float * in, const int nx, const int ny) 
  {
      for (int iy = 0; iy < ny; ++iy) 
      {
          for (int ix = 0; ix < nx; ++ix) 
          {
              out[ix * ny + iy] = in[iy * nx + ix];
          }
      }
  }
  ```
  - **Reads**: Accessed by rows in the original matrix; results in coalesced access. 
  - **Writes**: Accessed by columns in the transposed matrix; results in strided access. 
- Setting An Upper and Lower Performance Bound for Transpose Kernels. 
  - Copy the matrix by loading and storing rows (upper bound). 
    - This simulates the transpose but with only coalesced accesses. 
  - Copy the matrix by loading and storing columns (lower bound). 
    - This simulates the transpose but with only strided accesses.
```c++
__global__ void copyRow(float * out, const float * in, const int nx, const int ny) 
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) 
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__ void copyCol(float * out, const float * in, const int nx, const int ny) 
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) 
    {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}
```
- Naive Transpose: Reading Rows versus Reading Columns
  - For the `NaiveCol` implementation, store requests are never replayed due to coalesced writes, 
    but load requests are replayed many times due to strided reads. 
  - Caching of loads in L1 cache can limit the negative performance impact of strided loads. 
```c++
/// Loads by row and stores by column. 
__global__ void transposeNaiveRow(float * out, const float * in, const int nx, const int ny) 
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) 
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

/// Loads by column and stores by row. 
/// Performs better on loading with L1-cache enabled. 
///
/// Xi:
/// Suppose loads and stores are almost equally expensive (NOT TRUE in all scenarios!). 
/// NaiveRow, Load-by-row and store-by-column: 
///     Loading is colaesced, no extra overhead. 
///     Storing is strided, full extra overhead. 
/// NaiveCol, Load-by-column and store-by-row: 
///     Loading is strided, (in concept) full extra overhead, 
///         BUT one warp brings into L1 cache the data needed for neighboring warps (their loads), 
///         (note that warps are colaesced w.r.t. threadIdx, and that
///         all warps in one thread block reside on one unique SM, 
///         sharing the same on-chip L1-cache/shared-memory block), 
///         thus achieves sub-full extra overhead;
///     Storing is colaesced, no extra overhead. 
/// Overall: 
///     NaiveCol performs better, as it utilizes the L1-cache 
///     to negate the overhead of strided memory access patterns, 
///     which is available only to loading operations. 
///     So it's better to make loads strided (negated by L1-cache) 
///     and stores coalesced (no extra overhead). 
/// 
/// Textbook: 
/// While the reads performed by column will be uncoalesced 
/// (hence bandwidth will be wasted on bytes that were not requested), 
/// bringing those extra bytes into the L1 cache means that 
/// the next read may be serviced out of cache rather than global memory. 
/// Global memory writes does not go through L1-cache, no difference on performance. 
__global__ void transposeNaiveCol(float * out, const float * in, const int nx, const int ny) 
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) 
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}
```
```
L1-cache Enabled: 

Effective Bandwidth: 
CopyRow:  125.67 GB/s
CopyCol:   58.76 GB/s
NaiveRow:  64.16 GB/s
NaiveCol:  81.64 GB/s

Throughput and Efficiency: 
CopyRow:  gld_throughput 131.46 GB/s gst_throughput  65.32 GB/s gld_efficiency 49.81% gst_efficiency 100.00%
CopyRow:  gld_throughput 475.67 GB/s gst_throughput 118.52 GB/s gld_efficiency  6.23% gst_efficiency  25.00%
NaiveRow: gld_throughput 129.05 GB/s gst_throughput  64.31 GB/s gld_efficiency 50.00% gst_efficiency  25.00%
NaiveCol: gld_throughput 642.33 GB/s gst_throughput  40.02 GB/s gld_efficiency  6.21% gst_efficiency 100.00%
```
- Unrolling Transpose
  - Assign more independent work to each thread to maximize in-flight memory requests.
```c++
/// Load by row and store by column
__global__ void transposeUnroll4Row(float * out, const float * in, const int nx, const int ny) 
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;  // access in rows
    unsigned int to = ix * ny + iy;  // access in columns
    
    if (ix + 3 * blockDim.x < nx && iy < ny) 
    {
        out[to] = in[ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

/// Load by column and store by row
__global__ void transposeUnroll4Col(float * out, const float * in, const int nx, const int ny) 
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;  // access in rows
    unsigned int to = ix * ny + iy;  // access in columns
    
    if (ix + 3 * blockDim.x < nx && iy < ny) 
    {
        out[ti] = in[to];
        out[ti + blockDim.x] = in[to + blockDim.x * ny];
        out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
        out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
    }
}
```
- Tested on NVIDIA GeForce RTX 2080 Ti, with `nx = ny = 1 << 11`. 
  - Profile a kernel function named `myKernelFunc` from executable `./myProgramExecutable` 
    - `$ ncu -k regex:myKernelFunc ./myProgramExecutable [command-line arguments...]`
    - `ncu` could not profile time cost because of kernel replays!
  - Note that a better memory footprint does **not** equal to better performance (overall time taken)!
  - [NVIDIA Development Tools Solutions - ERR_NVGPUCTRPERM: Permission issue with Performance Counters](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
```bash
# These Nsight Compute metrics "translated" into legacy nvprof metrics: 
# gld_throughput, gst_throughput, gld_efficiency, gst_efficiency
ncu -k regex:myKernelFunc --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./myProgramExecutable
```
```
----------------------- ---- -------- -------- ---------- ----------
Metric                  Unit NaiveRow NaiveCol Unroll4Row Unroll4Col
----------------------- ---- -------- -------- ---------- ----------
Time Elapsed              ms    0.093    0.087      0.078      0.075
Effective Bandwidth     GB/s   359.04   384.21     426.27     445.37
Global  Load Throughput GB/s   192.12   839.87     227.06     967.32
Global Store Throughput GB/s   768.47   209.97     912.20     241.83
Global  Load Efficiency    %      100       25        100         25
Global Store Efficiency    %       25      100         25        100
----------------------- ---- -------- -------- ---------- ----------

P.S. Original data collected for nx = ny = 1 << 11: 
transposeNaiveRow<<<grid (128x128), block (16x16)>>> ran for 0.093456ms, effective bandwidth 359.0398858364917 GB/s
transposeNaiveCol<<<grid (128x128), block (16x16)>>> ran for 0.087334ms, effective bandwidth 384.208106545478 GB/s
transposeUnroll4Row<<<grid (32x128), block (16x16)>>> ran for 0.078717ms, effective bandwidth 426.26665538018864 GB/s
transposeUnroll4Col<<<grid (32x128), block (16x16)>>> ran for 0.075341ms, effective bandwidth 445.36749747338024 GB/s

When nx = ny = 1 << 13, the results are inverted!
transposeNaiveRow<<<grid (512x512), block (16x16)>>> ran for 1.6378ms, effective bandwidth 327.7996449420446 GB/s
transposeNaiveCol<<<grid (512x512), block (16x16)>>> ran for 2.1792ms, effective bandwidth 246.3616982564283 GB/s
transposeUnroll4Row<<<grid (128x512), block (16x16)>>> ran for 1.76699ms, effective bandwidth 303.8344394371926 GB/s
transposeUnroll4Col<<<grid (128x512), block (16x16)>>> ran for 2.21361ms, effective bandwidth 242.53162715341088 GB/s
```
- Diagonal Transpose
  - Physically, all thread blocks are arranged in 1D. 
    - Logical layout could be 1D, 2D, or 3D (`dim3 gridDim`). 
    - Each block has its identifier;
      - E.g., Cartesian coordinate system (row-major);
      - `int bid = blockIdx.y * blockDim.x + blockIdx.x;`. 
  - **No** direct control over the order in which thread blocks are scheduled. 
    - DRAM: Memory access serviced by 256-Byte partitions
    - Cartesian coordinate
      - Access not evenly distributed among partitions (*partition camping*)
        - Some partitions are accessed more than once (queued)
        - Other partitions are never accessed (idle)
    - Diagonal block-coordinate system
      - Nonlinear mapping
      - Strided accesses are not likely to fall into a single partition
  - Note: 
    - Thread blocks are "shuffled"
      - Adjacent thread blocks might not be assigned to adjacent SMs
    - Warps in one thread block are "consecutive" (exploits spatial locality)
      - Wrap 0: threadIdx 0-31
      - Warp 1: threadIdx 32-63
      - ...
- Thin Blocks:
  - Block Dim $(16, 16) \to (8, 32)$
  - A thin block improves the effectiveness of store operations
  - Increases the number of consecutive elements stored by a thread block
```
transposeNaiveRow<<<grid (128x128), block (16x16)>>> ran for 0.091892ms, effective bandwidth 365.1507512092593 GB/s
transposeNaiveRow<<<grid (256x64), block (8x32)>>> ran for 0.073047ms, effective bandwidth 459.35401567461935 GB/s

transposeNaiveCol<<<grid (128x128), block (16x16)>>> ran for 0.09071ms, effective bandwidth 369.9088663155546 GB/s
transposeNaiveCol<<<grid (256x64), block (8x32)>>> ran for 0.074991ms, effective bandwidth 447.4461010914167 GB/s
```

### 🎯 MATRIX ADDITION WITH UNIFIED MEMORY

- Manual:
```c++
void initializeData(float * data, const int size)
{
    static std::default_random_engine e(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    static std::normal_distribution<float> n(0.0f);

    for (int i = 0; i < size; i++)
    {
        data[i] = n(e);
    }
}
```
```c++
constexpr int kNx {1 << 11};
constexpr int kNy {1 << 11};
constexpr int kMatrixSize {kNx * kNy};
constexpr int kNBytes {sizeof(float) * kMatrixSize};

std::vector<float> A(kNx * kNy);
std::vector<float> B(kNx * kNy);
std::vector<float> C(kNx * kNy);

initializeData(A.data(), kNx * kNy);
initializeData(A.data(), kNx * kNy);

float * dA;
float * dB;
float * dC;

cudaMalloc(reinterpret_cast<void **>(&dA), kNBytes);
cudaMalloc(reinterpret_cast<void **>(&dB), kNBytes);
cudaMalloc(reinterpret_cast<void **>(&dC), kNBytes);

cudaMemcpy(dA, A.data(), kNBytes, cudaMemcpyHostToDevice);
cudaMemcpy(dB, B.data(), kNBytes, cudaMemcpyHostToDevice);

// Warm-up kernel for accurate timing results. 
sum<<<grid, block>>>(dA, dB, dC, 1, 1);

sum<<<grid, block>>>(dA, dB, dC, kNx, kNy);
cudaMemcpy(c.data(), dC, kNBytes, cudaMemcpyDeviceToHost);

// Dereference and use C on host (after cudaMemcpy)
// ...

cudaFree(dA);
cudaFree(dB);
cudaFree(dC);
```
- Managed memory:
  - On Geforce RTX 2080 Ti:
    - Overall performance **degrade** by 100%! Takes 2x time. 
      - Time from memory allocation to free and device reset. 
  - P.S. On Kepler K40 (Data from textbook):
    - Takes longer to initialize data. 
      - Data are initially allocated on GPU;
      - Copied to CPU for initialization; not needed in the manual version. 
    - Warm up kernel: 
      - **IMPORTANT**! 
      - Brings CPU data back to GPU. 
      - If omitted, kernel with managed memory will run significantly slower!
    - No Explicit memcpys. 
    - Kernel launchs faster than the manual version. 
    - Kernel time nearly the same as the manual version. 
```c++
float * A;
float * B;
float * C;

cudaMallocManaged(reinterpret_cast<void **>(&A), kNBytes);
cudaMallocManaged(reinterpret_cast<void **>(&B), kNBytes);
cudaMallocManaged(reinterpret_cast<void **>(&C), kNBytes);

initializeData(A, kMatrixSize);
initializeData(B, kMatrixSize);

// Warm-up kernel. IMPORTANT FOR PERFORMANCE! 
// With unified memory, all pages will migrate from host to device. 
sum<<<grid, block>>>(dA, dB, dC, 1, 1);

sum<<<grid, block>>>(A, B, C, kNx, kNy);
cudaDeviceSynchronize();

// Dereference and use C directly on host
// ...

cudaFree(A);
cudaFree(B);
cudaFree(C);
```



## 🌱 5 Shared Memory and Constant Memory

### 🎯 INTRODUCING CUDA SHARED MEMORY

#### 📌 Shared Memory (SMEM)

- Shared memory (SMEM)
  - Shared by all threads in the thread block (currently executing on that SM);
    - A fixed amount of SMEM is allocated to each thread block when it starts executing;
    - This SMEM address space is shared by all threads in this thread block;
    - Accesses issued per warp. 
      - Ideally: Each request by a warp is serviced in one transaction; 
      - Worst case: Sequentially in 32 unique transactions.  
        - All threads access the same word in SMEM
        - One fetches it and multicast it to other threads
  - Latency: Roughly 20-30x lower than global memory;
  - Bandwidth: Nearly 10x higher. 
- PROGRAM-MANAGED CACHE
  - C Programming (CPU cache)
    - Transparent to C program, **no** control over it
    - Could only tune the algorithm (iteration order)
  - CUDA Shared Memory (SMEM)
    - Full control over it
      - When data is moved into and when evicted

#### 📌 Shared Memory Allocation

- Static Allocation
  - Known size
    - Decalred inside a kernel function: Local to that kernel
    - Declared outside of any kernels, in a file: Global to all kernels
  - 1D, 2D or 3D
  ```c++
  __shared__ float tile[kNy][kNx];
  ```
- Dynamic Allocation
  - Unknown size (*unsized array*) 
  - Declare with `extern` keyword
  - Allocate dynamically when launching kernels (pass in size **in Bytes**)
  - *Only 1D* supported
  ```c++
  extern __shared__ float tile[];
  // ...
  kernel<<<grid, block, kTileSize * sizeof(float)>>>(arguments...);
  ```

#### 📌 Shared Memory Banks and Access Mode

- Shared Memory could hide the impact of global memory latency and bandwidth
  - Two key properties to mesure when optimizing memory performance
    - Latency
    - Bandwidth
- [Memory **Banks**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x)
  - SMEM: 1D memory space
  - SMEM is divided into 32 equally-sized memory modules (*banks*)
    - Strided segmentation: [Successive 32-bit words map to successive banks](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x). 
      - Addresses 0-31x4 * fall into bank 0-31, 
      - Addresses 32x4-63x4 again fall into bank 0-31, 
      - ...
    - Banks could be accessed simultaneously
    - We have 32 banks because a warp has 32 threads
  - Number of transactions of a warp's SMEM access request
    - 1 transaction if accessing no more than one memory location per bank
    - 2/2+ transactions otherwise
- **Bank Conflict**
  - When?
    - When multiple addresses in a SMEM request fall into the same bank
    - Splits the request into separate conflict-free serial transactions
  - Three Types
    - **Parallel Access**
      - Multiple addresses accessed across multiple banks
      - Most common
      - One or more transactions depending on conflicts
    - **Serial Access**
      - Multiple addresses fall into the same bank
      - Wrost pattern, request must be serialized
        - 32 serial transactions if all threads in a warp access the same bank
    - **Broadcast Access**
      - All threads in a warp read a single address (in a single bank)
      - One memory transaction, result broadcasted to all threads
      - Poor bandwidth footprint
  - When several threads access the same bank:
    - Conflict-free broadcast access when accessing the same address
    - Bank conflict access when accessing different addresses within a bank
- Access Mode
  - Memory bank width: 32-bit (one word)
    - Successive 32-bit words map to successive bank
    - Bandwidth: 32 bits per clock cycle.
- Memory Padding
  - Add padding words
    - Suppose 5 banks 01234, append a padding x "in the last column (logically)"
    - 5-way conflict on bank 0 turns into regular parallel access
  ```
  01234x    01234
  01234x    x0123
  01234x => 4x012
  01234x    34x01
  01234x    0234x
  ```
  - Padded memory is never used for data storage

#### 📌 Configuring the Amount of Shared Memory

- Turing architecture (compute capability 7.5)
  - Unified data cache has a size of 96KB
  - SMEM capacity can be set to either 32KB or 64KB
  - The driver automatically configures the SMEM capacity *for each kernel*
    - Avoids SMEM occupancy bottlenecks 
    - Allows concurrent execution with already-launched kernels where possible. 
    - In most cases, the driver's default behavior should provide optimal performance.
- Applications can provide additional hints regarding the desired shared memory configuration
  - [cudaFuncSetCacheConfig](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g422642bfa0c035a590e4c43ff7c11f8d)
  ```c++
  template <class T>
  inline __host__ ​cudaError_t cudaFuncSetCacheConfig (T * func, cudaFuncCache cacheConfig);

  // cudaFuncCachePreferNone = 0    Default function cache configuration, no preference
  // cudaFuncCachePreferShared = 1  Prefer larger shared memory and smaller L1 cache
  // cudaFuncCachePreferL1 = 2      Prefer larger L1 cache and smaller shared memory
  // cudaFuncCachePreferEqual = 3   Prefer equal size L1 cache and shared memory
  ```
  - [cudaFuncSetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g422642bfa0c035a590e4c43ff7c11f8d)
  - Where a chosen integer percentage does not map exactly to a supported capacity, the next larger capacity is used.
    - Supported in Compute Compatibility 7.x: 0, 8, 16, 32, 64, or 96KB;
    - For instance, in the example above, 50% of the 96KB maximum is 48KB, 
      - which is not a supported shared memory capacity. 
      - Thus, the preference is rounded up to 64KB.
  ```c++
  template <class T>
  inline __host__ ​cudaError_t cudaFuncSetAttribute(T * entry, cudaFuncAttribute attr, int value);

  // Device code
  __global__ void myKernel(...)
  {
      __shared__ float buffer[kBlockDim];
      ...
  }

  // Host code
  // Prefer shared memory capacity 50% of maximum
  int carveout = 50;   

  // Named Carveout Values:
  // carveout = cudaSharedmemCarveoutDefault;   //  (-1)
  // carveout = cudaSharedmemCarveoutMaxL1;     //   (0)
  // carveout = cudaSharedmemCarveoutMaxShared; // (100)             

  cudaFuncSetAttribute(myKernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
  myKernel<<<grid, block>>>(...);
  ```
  - A single thread block can address the full capacity of shared memory (64KB on Turing). 
    - Kernels relying on shared memory allocations over 48KB per block are architecture-specific, 
      - as such they must use dynamic shared memory (rather than statically sized arrays)
      - require an explicit opt-in using `cudaFuncSetAttribute` as follows.
  ```c++
  // Device code
  __global__ void myKernel(...)
  {
      extern __shared__ float buffer[];
      ...
  }

  // Host code
  constexpr int kMaxBytes = 64 * 1024;  // 64KB
  cudaFuncSetAttribute(myKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxBytes);
  kernel<<<grid, block, kMaxBytes>>>(...);
  ```
- Synchronization
  - CUDA runtime functions for intra-block synchronization: 
    - Barriers
      - Calling threads wait for all other calling threads to reach the barrier point. 
    - Memory fences
      - Calling threads stall until all modifications to memory are visible to all other calling threads.
  - CUDA's Weakly-Ordered Memory Model
    - Order in which GPU thread writes could be **different** from the order in which they appear in *source code*. 
      - Shared memory
      - Global memory
      - Page-locked host memory
      - Memory of a peer device
    - Order in which GPU thread reads could be **different** from the *source code*
      - If instructions are independent of each other
    - Barriers and fences are necessary to explicitly enforce ordering
  - [Synchronization Functions (Explicit Barriers)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)
    - Only possible to perform a barrier among threads in the same thread block
    ```c++
    /// Waits until 
    /// (1) all threads in the thread block have reached this point
    /// and 
    /// (2) all global and shared memory accesses made by these threads prior to __syncthreads() 
    ///     are visible to all threads in the block. 
    void __syncthreads();
    ```
    - Thread blocks can be executed in any order
      - In parallel or in series
      - On any SM
    - Global synchronization across blocks could be "simulated"
      - Split the kernel apart at the synchronizatio point;
      - Perform multiple kernel launches. 
      - Each successive kernel launch must wait for the preceding kernel launch to complete
        - Implicit global barrier
  - Be careful when using `__syncthreads` in conditional code
    - Valid only if a conditional is guaranteed to evaluate identically across the entire thread block;
    - Otherwise: Hangs execution or unintended side effects
    ```c++
    // May cause threads in a block to wait indefinitely for each other 
    // because all threads in a block never hit the same barrier point. 
    if (threadID % 2 == 0)
    {
        __syncthreads();
    }
    else
    {
        __syncthreads();
    }
    ```
  - [Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
    - Enforce a sequentially-consistent ordering on memory accesses for all threads inside a thread block
    - Memory fence functions only affect the ordering of memory operations by a thread.
      - They do **not**, by themselves, ensure that these memory operations are visible to other threads!
    - Block-level
      - Equivalent to [cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html)
      ```c++
      // Ensures that 
      // all writes to shared/global memory made by a calling thread before the fence 
      // are visible to other threads in the same block after the fence. 
      void __threadfence_block();
      ```
    - Grid-level
      - Equivalent to [cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device)](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html)
      ```c++
      // Stalls the calling thread until 
      // all of its writes to global memory are visible to all threads in the same grid. 
      void __threadfence();
      ```
    - System-level (including host and device)
      - Equivalent to [cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system)](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html)
      ```c++
      // Stalls the calling thread to ensure 
      // all its writes to global memory, pagelocked host memory, and the memory of other devices 
      // are visible to all threads in all devices and host threads.
      void __threadfence_system();
      ```
    - E.g. Thread 1 executes `write` and thread 2 executes `read`
      - 4 possible results without fences
        - `a == 1 && b == 2`
        - `a == 1 && b == 20`
        - `a == 10 && b == 2`
        - `a == 10 && b == 20`
      ```c++
      __device__ int x = 1, y = 2;

      __device__ write()
      {
          // Write to x does NOT necessarily happen before write to y
          x = 10;
          y = 20;
      }

      __device__ read()
      {
          // Write to b does NOT necessarily happen before write to a
          int b = y;
          int a = x;
      }
      ```
      - 3 possible results remains with grid-level fences
        - If `b` is assigned with the updated `y`, then `a` is guaranteed to be assigned with the updated `x`;
        - If thread 1 and 2 belong to the same block, it is enough to use `__threadfence_block()`;
        - `__threadfence()` must be used if they are CUDA threads from the same device;
        - `__threadfence_system()` must be used if they are CUDA threads from two different devices.
      ```c++
      __device__ int x = 1, y = 2;

      __device__ write()
      {
          // Write to x is guaranteed to happen before write to y
          x = 10;
          __threadfence();
          y = 20;
      }

      __device__ read()
      {
          // Write to b is guaranteed to happen before write to a
          int b = y;
          __threadfence();
          int a = x;
      }
      ```
  - Volatile Qualifier
    - For variables in global or shared memory;
    - Enforce cache-free direct stores/loads to memory upon these variables;
      - GPU assume these variables can be changed/used at any time by any other thread. 
- **SHARED MEMORY VERSUS GLOBAL MEMORY**
  - SMEM
    - On-chip (cache)
    - 20 to 30 times lower latency than DRAM
    - Greater than 10 times higher bandwidth than DRAM
    - Smaller access granularity
  - Global Memory
    - On device memory (DRAM)

### 🎯 CHECKING THE DATA LAYOUT OF SHARED MEMORY

- Topics:
  - Square versus rectangular arrays
  - Row-major versus column-major accesses
  - Static versus dynamic shared memory declarations
  - File-scope versus kernel-scope shared memory
  - Memory padding versus no memory padding
- Considerations for SMEM programming
  - Mapping data elements across memory banks
  - Mapping from thread index to shared memory offset

#### 📌 Square Shared Memory

- A shared memory tile with 32 elements in each dimension (row-major). 

|    **Byte Address**   | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | ... | 4088 | 4092 |
|:---------------------:|:-:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|:---:|:----:|:----:|
| **4-Byte Word Index** | 0 | 1 | 2 |  3 |  4 |  5 |  6 |  7 | ... | 1022 | 1023 |

|            | **Bank 0** | **Bank 1** | **Bank 2** | **...** | **Bank 31** |
|:----------:|:----------:|:----------:|:----------:|:-------:|:-----------:|
|  **Row 0** |      0     |      1     |      2     |   ...   |      31     |
|  **Row 1** |     32     |     33     |     34     |   ...   |      63     |
|  **Row 2** |     64     |     65     |     66     |   ...   |      95     |
|   **...**  |     ...    |     ...    |     ...    |   ...   |     ...     |
| **Row 31** |     992    |     993    |     994    |   ...   |     1023    |
|            |  **Col 0** |  **Col 1** |  **Col 2** | **...** |  **Col 31** |

```c++
// Consider a grid with only one block of size (32, 32). 
constexpr dim3 kGridDim {1U};
constexpr dim3 kBlockDim {32U, 32U};
__shared__ int tile[kBlockDim.x][kBlockDim.y];

// Note that index dim 0 is row (corrsponding to y), 
// and index dim 1 is column (corresponding to x)! 
tile[threadIdx.y][threadIdx.x]  // Parallel pattern, no confict!
tile[threadIdx.x][threadIdx.y]  // Bank conflict!
```
- Accessing Row-Major versus Column-Major
  - setRowReadRow: No conflict
  - setColReadCol: 16-way conflict
  - [Local Test](./examples/pccp/pccp_220_smem_bank_xy_vs_yx.cu)
```c++
__global__
void setRowReadRow(int * __restrict__ out)
{
    __shared__ int tile[kBlockDim.x][kBlockDim.y];
    
    auto idx = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__
void setColReadCol(int * __restrict__ out)
{
    __shared__ int tile[kBlockDim.x][kBlockDim.y];

    auto idx = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}
```
```
$ nvprof ./cmake-build-release/cumo

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   40.52%  3.9680us         1  3.9680us  3.9680us  3.9680us  setColReadCol(int*)
                   29.08%  2.8480us         1  2.8480us  2.8480us  2.8480us  setRowReadRow(int*)

$ ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum ./cmake-build-release/cumo

  setRowReadRow(int *)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                                               32
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                                               32
    ---------------------------------------------------------------------- --------------- ------------------------------

  setColReadCol(int *)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                                            1,024
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                                            1,024
    ---------------------------------------------------------------------- --------------- ------------------------------
```
- Writing Row-Major and Reading Column-Major
  - Store: conflict-free;
  - Load: 16-way conflict.
```c++
__global__
void setRowReadCol(int * __restrict__ out)
{
    __shared__ int tile[kBlockDim.x][kBlockDim.y];

    auto idx = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

void setRowReadColDyn(int * __restrict__ out)
{
    extern __shared__ int tile[];

    auto idx = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    tile[threadIdx.y * blockDim.x + threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x * blockDim.y + threadIdx.y];
}
```
```
$ ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum ./cmake-build-release/cumo

  setRowReadCol(int *)
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                                            1,024
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                                               32
    ---------------------------------------------------------------------- --------------- ------------------------------
```
- Padding Shared Memory
  - Pad one element in each row
  - Column elements are distributed among different banks
  - Both reading and writing operations are conflict-free.
```c++
constexpr int kPad {1};

__global__
void setRowReadColPad(int * __restrict__ out)
{
    __shared__ int tile[kBlockDim.x][kBlockDim.y + kPad];

    auto idx = static_cast<int>(threadIdx.y * blockDim.x + blockDim.x);
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__
void setRowReadColDynPad(int * __restrict__ out)
{
    extern __shared__ int tile[];

    auto idx = static_cast<int>(threadIdx.y * blockDim.x + blockDim.x);
    tile[threadIdx.y * (blockDim.x + kPad) + threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x * (blockDim.x + kPad) + threadIdx.y];
}
```

### 🎯 REDUCING GLOBAL MEMORY ACCESS

- Baseline Parallel Reduction
```c++
// unroll4 + complete unroll for loop + gmem
__global__ void reduceGmem(int * g_idata, int * g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    int * idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();
    if (blockDim.x >=  512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();
    if (blockDim.x >=  256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();
    if (blockDim.x >=  128 && tid <  64) idata[tid] += idata[tid +  64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmemDyn(int * g_idata, int * g_odata, unsigned int n)
{
    extern __shared__ int smem[];

    // set thread ID
    unsigned int tid = threadIdx.x;
    int * idata = g_idata + blockIdx.x * blockDim.x;

    // set to smem by each threads
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >=  512 && tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >=  256 && tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >=  128 && tid <  64) smem[tid] += smem[tid +  64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int * vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0)  g_odata[blockIdx.x] = smem[0];
}
```
```
reduce at device 0: Tesla K40c with array size 16777216 grid 131072 block 128
Time(%)  Time      Calls  Avg       Min       Max       Name
2.01%    2.1206ms  1      2.1206ms  2.1206ms  2.1206ms  reduceGmem()
1.10%    1.1536ms  1      1.1536ms  1.1536ms  1.1536ms  reduceSmem()
```
- Parallel Reduction with Unrolling
  - Unroll blocks to improve kernel performance by enabling multiple I/O operations to be in-flight at once.
  - Increased global memory throughput by exposing more parallel I/O per thread
  - Reduction of global memory store transactions by one-fourth
  - Overall kernel performance improvement
```c++
__global__ void reduceSmemUnrollDyn(int * g_idata, int * g_odata, unsigned int n)
{
    extern __shared__ int smem[];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling 4
    int tmpSum = 0;

    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        tmpSum = a1 + a2 + a3 + a4;
    }

    smem[tid] = tmpSum;
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >=  512 && tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >=  256 && tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >=  128 && tid <  64) smem[tid] += smem[tid +  64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int * vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}
```

### 🎯 COALESCING GLOBAL MEMORY ACCESSES

```c++
// Baseline
__global__ void naiveGmem(float * out, float * in, const int nx, const int ny)
{
    // matrix coordinate (ix,iy)
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeSmemDyn(float * out, float * in, int nx, int ny)
{
    // dynamic shared memory
    extern __shared__ float tile[];

    // coordinate in original matrix
    unsigned int ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int row_idx, col_idx, irow, icol;
    row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    irow    = row_idx / blockDim.y;
    icol    = row_idx % blockDim.y;
    col_idx = icol * blockDim.x + irow;

    // coordinate in transposed matrix
    ix = blockDim.y * blockIdx.y + icol;
    iy = blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        // load data from global memory to shared memory
        tile[row_idx] = in[ti];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[to] = tile[col_idx];
    }
}

__global__ void transposeSmemPadDyn(float * out, float * in, int nx, int ny)
{
    // static shared memory with padding
    extern __shared__ float tile[];

    // coordinate in original matrix
    unsigned int ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int idx     = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int row_idx = threadIdx.y * (blockDim.x + kPad) + threadIdx.x;
    unsigned int irow    = idx / blockDim.y;
    unsigned int icol    = idx % blockDim.y;
    unsigned int col_idx = icol * (blockDim.x + kPad) + irow;

    // coordinate in transposed matrix
    ix = blockDim.y * blockIdx.y + icol;
    iy = blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        // load data from global memory to shared memory
        tile[row_idx] = in[ti];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[to] = tile[col_idx];
    }
}

__global__ void transposeSmemUnrollPad(float * out, float * in, const int nx, const int ny) 
{
    // static 1D shared memory with padding
    __shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];

    // coordinate in original matrix
    unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // linear global memory index for original matrix
    unsigned int ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    unsigned int ix2 = blockIdx.y * blockDim.y + icol;
    unsigned int iy2 = 2 * blockIdx.x * blockDim.x + irow;

    // linear global memory index for transposed matrix
    unsigned int to = iy2 * ny + ix2;

    if (ix + blockDim.x < nx && iy < ny)
    {
        // load two rows from global memory to shared memory
        unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;
        tile[row_idx] = in[ti];
        tile[row_idx+BDIMX] = in[ti+BDIMX];

        // thread synchronization
        __syncthreads();

        // store two rows to global memory from two columns of shared memory
        unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}
```

### 🎯 CONSTANT MEMORY

- Constant Memory
  - Read/write permissions
    - read-only from kernel code
    - readable and writable from the host
  - Resides in device DRAM (like global memory), but has dedicated on-chip cache (like L1 cache & smem)
    - Per-SM constant cache faster than constant memory. 
    - 64 KB constant memory cache per SM.
  - Optimal access pattern: 
    - All threads in a warp access the same location in constant memory. 
    - Accesses to different addresses within a warp are serialized. 
  - Constant memory variables: `__constant__ T var;`
    - Exist for the lifespan of the application;
    - Accessible from all threads within a grid;
    - Accessible from the host through runtime functions;
    - Visible across multiple source files when using CUDA separate compilation. 
  - Values must be initialized from host: [cudaMemcpyToSymbol](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9bcf02b53644eee2bef9983d807084c7)
  ```c++
  __host__ 
  ​cudaError_t cudaMemcpyToSymbol(
        const void * symbol, 
        const void * src, 
        size_t count, 
        size_t offset = 0, 
        cudaMemcpyKind kind = cudaMemcpyHostToDevice
  );
  ```

#### 📌 Implementing a 1D Stencil with Constant Memory

A 9-point stencil for 1D 1-st order derivative: 
$$
f^{'}(x) \approx 
c_0 (f(x + 4h) - f(x - 4h)) + 
c_1 (f(x + 3h) - f(x - 3h)) + 
c_2 (f(x + 2h) - f(x - 2h)) + 
c_3 (f(x + h) - f(x - h))
. 
$$
```c++
#define RADIUS 4
#define BDIM 32

__constant__ float coef[RADIUS + 1];

__global__ void stencil_1d(float * in, float * out, int N)
{
    // RADIUS defines the number of points on either side of a point x that are used to calculate its value.
    // For this example, RADIUS is defined as 4 to form a 9-point stencil. 
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < N)
    {
        // index to shared memory for stencil calculation
        int sidx = threadIdx.x + RADIUS;

        // Read data from global memory into shared memory
        smem[sidx] = in[idx];

        // read halo part to shared memory
        if (threadIdx.x < RADIUS)
        {
            smem[sidx - RADIUS] = in[idx - RADIUS];
            smem[sidx + BDIM] = in[idx + BDIM];
        }

        // Synchronize (ensure all the data is available)
        __syncthreads();

        // Apply the stencil
        float tmp = 0.0f;

#pragma unroll
        for (int i = 1; i <= RADIUS; i++)
        {
            tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
        }

        // Store the result
        out[idx] = tmp;

        // Loop while idx < n in case gridSize * blockSize insufficient to cover all samples
        idx += gridDim.x * blockDim.x;
    }
}
```

#### 📌 Comparing with the Read-Only Cache

- Use GPU texture pipeline as a read-only cache for global memory. 
  - Separate read-only cache with separate memory bandwidth from normal global memory reads
  - Boosts performance for bandwidth-limited kernels.
- Two ways to access global memory through the read-only cache: 
  - Intrinsic function `__ldg`
    - Used in place of a normal pointer dereference
    - Force a load to go through the read-only data cache
  ```c++
  __global__ void kernel(float * output, float * input)
  {
      // ...
      output[idx] += __ldg(&input[idx]);
      // ...
  }
  ```
  - Qualify pointers to global memory with `const T * __restrict__`
  ```c++
  void kernel(float* output, const float* __restrict__ input)
  {
      // ...
      output[idx] += input[idx];
      // ...
  }
  ```
- Intrinsic `__ldg` is a better choice when: 
  - More explicit control over the read-only cache mechanism is desired; or
  - Code is so complex that the compiler may be unable to detect that read-only cache use is safe.
- Read-only cache is separate and distinct from constant cache. 
  - Data loaded through constant cache:
    - Must be relatively small, and
    - Must be accessed uniformly for good performance. 
      - All threads of a warp should access the same location at any given time. 
  - Data loaded through read-only cache: 
    - Can be much larger, and 
    - Can be accessed in a non-uniform pattern.
- **CONSTANT CACHE VERSUS READ-ONLY CACHE**
  - Both are read-only from the device;
  - Both have limited per-SM resources: 
    - Constant cache: 64 KB,
    - Read-only cache: 48 KB.
  - Constant cache performs better on uniform reads:  
    - where every thread in a warp accesses the same address;
  - Read-only cache is better for scattered reads. 

### 🎯 THE WARP SHUFFLE INSTRUCTION

- *Shuffle* Instruction 
  - Allows threads to directly read registers of other threads in the same warp;
  - Lower latency than shared memory;
    - **NOT** much faster than a single, non-bank-conflicted shared memory read. 
  - Does not consume extra memory to perform a data exchange. 
- *Lane*
  - A lane is simply a single thread within a warp. 
  - Each lane: Uniquely identified by its lane index $\in [0, 31]$. 
  ```c++
  laneIdx = threadIdx.x % 32
  warpIdx = threadIdx.x / 32
  ```

#### 📌 Variants of the Warp Shuffle Instruction

- [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
```c++
/// Copy from thread srcLane % width. 
T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize);

/// Copy from thread (callerLane - delta) % width. 
/// Lower delta lanes will be unchanged if width < warpSize. 
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width = warpSize);

/// Copy from thread (callerLane + delta) % width. 
/// Upper delta lanes will be unchanged if width < warpSize. 
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = warpSize);

/// Copy from thread (callerLane xor laneMask) % width. 
/// Each group of width consecutive threads are able to access elements from earlier groups of threads. 
/// If they attempt to access elements from later groups of threads, their own value of var will be returned. 
/// Implements a butterfly addressing pattern such as is used in tree reduction and broadcast.
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);
```
- Permit exchanging of a variable between threads within a warp without use of shared memory. 
  - Occurs simultaneously for all active threads within the warp (and named in mask);
  - Moves 4 or 8 Bytes of data per thread depending on the type.
- Threads may only read data from another thread which is actively participating in the `__shfl_sync()` command. 
  - If the target thread is inactive, the retrieved value is undefined.
  - See `mask` (below). 
- Do **not** imply a memory barrier. Do **not** guarantee memory ordering.
- `T` can be:
  - `int`, `unsigned int`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float` or `double`; 
  - `__half` or `__half2` (`#include <cuda_fp16.h>`);
  - `__nv_bfloat16` or `__nv_bfloat162` (`#include <cuda_bf16.h>`).
- `mask`
  - Indicates threads participating in the call. 
  - Bits representing particitipating threads' lane IDs must be set, otherwise is *undefined behavior*. 
    - Each calling thread must have its own bit set in the mask, and: 
    - All non-exited threads named in mask must execute the same intrinsic with the same mask. 
- `width` 
  - Must have a value which is a power of two $\le \mathrm{warpSize}$ (i.e., 1, 2, 4, 8, 16 or 32). 
  - Results are *undefined* for other values.

#### 📌 Sharing Data within a Warp

- Broadcast of a Value across a Warp
```c++
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


constexpr dim3 kBlockDim {16U, 1U, 1U};
constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};
constexpr unsigned int kShflMask {0xFFFFFFFFU};


__global__ void test_shfl_broadcast(int * __restrict__ d_out, const int * __restrict__ d_in, int srcLane)
{
    int value = d_in[threadIdx.x];
    value = __shfl_sync(kShflMask, value, srcLane, kBlockDim.x);
    d_out[threadIdx.x] = value;
}


int main(int argc, char * argv[])
{
    thrust::device_vector<int> dIn(kBlockSize);
    thrust::sequence(thrust::device, dIn.begin(), dIn.end());
    thrust::device_vector<int> dOut(kBlockSize);

    test_shfl_broadcast<<<1U, kBlockDim>>>(
            dOut.begin().base().get(),
            dIn.begin().base().get(),
            2
    );
    cudaDeviceSynchronize();

    thrust::host_vector<int> hOut = dOut;

    for (int i : hOut)
        std::cout << i << ' ';
    std::cout << '\n';

    return EXIT_SUCCESS;
}
```
| init | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|:--:|
| shfl | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |  2 |  2 |  2 |  2 |  2 |  2 |
- Shift Up within a Warp
```c++
__global__ void test_shfl_up(int * __restrict__ d_out, const int * __restrict__ d_in, int delta = 2)
{
    int value = d_in[threadIdx.x];
    value = __shfl_up_sync(kShflMask, value, delta, kBlockDim.x);
    d_out[threadIdx.x] = value;
}
```
| init | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|:--:|
| shfl | 0 | 1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |  8 |  9 | 10 | 11 | 12 | 13 |
- Shift Down within a Warp
```c++
__global__ void test_shfl_down(int * __restrict__ d_out, const int * __restrict__ d_in, int delta = 2)
{
    int value = d_in[threadIdx.x];
    value = __shfl_down_sync(kShflMask, value, delta, kBlockDim.x);
    d_out[threadIdx.x] = value;
}
```
| init | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| shfl | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 14 | 15 |
- Shift within a warp with Wrap Around
```c++
__global__ void test_shfl_wrap(int * __restrict__ d_out, const int * __restrict__ d_in, int delta = 2)
{
    int value = d_in[threadIdx.x];
    value = __shfl_sync(kShflMask, value, threadIdx.x + delta, kBlockDim.x);
    d_out[threadIdx.x] = value;
}
```
| init | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| shfl | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |  0 |  1 |
- Butterfly Exchange across the Warp
```c++
__global__ void test_shfl_xor(int * __restrict__ dOut, const int * __restrict__ dIn, int laneMask = 0x1)
{
    int value = dIn[threadIdx.x];
    value = __shfl_xor_sync(kShflMask, value, laneMask, kBlockDim.x);
    dOut[threadIdx.x] = value;
}
```
| init | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|:--:|
| shfl | 1 | 0 | 2 | 3 | 5 | 4 | 7 | 6 | 9 | 8 | 11 | 10 | 13 | 12 | 15 | 14 |
- Exchange Values of an Array across a Warp
```c++
constexpr int kSegment {4};

__global__ void test_shfl_xor_array(int * __restrict__ dOut, const int * __restrict__ dIn, int laneMask = 0x1)
{
    int idx = threadIdx.x * kSegment;
    int value[kSegment];

    for (int i = 0; i < kSegment; i++) value[i] = dIn[idx + i];

    value[0] = __shfl_xor_sync(kShflMask, value[0], laneMask, kBlockDim.x);
    value[1] = __shfl_xor_sync(kShflMask, value[1], laneMask, kBlockDim.x);
    value[2] = __shfl_xor_sync(kShflMask, value[2], laneMask, kBlockDim.x);
    value[3] = __shfl_xor_sync(kShflMask, value[3], laneMask, kBlockDim.x);

    for (int i = 0; i < kSegment; i++) dOut[idx + i] = value[i];
}
```
| init | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| shfl | 4 | 5 | 6 | 7 | 0 | 1 | 2 | 3 | 12 | 13 | 14 | 15 |  8 |  9 | 10 | 11 |
- Exchange Values Using Array Indices Across a Warp
```c++
constexpr dim3 kBlockDim {16U, 1U, 1U};
constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};
constexpr unsigned int kShflMask {0xFFFFFFFFU};

constexpr int kSegment {4};

__inline__ __device__
void swap(int * __restrict__ value, int laneIdx, int laneMask, int firstIdx, int secondIdx)
{
    bool pred = ((laneIdx / laneMask + 1) == 1);

    if (pred)
    {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }

    value[secondIdx] = __shfl_xor_sync(kShflMask, value[secondIdx], laneMask, kBlockDim.x);

    if (pred)
    {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }
}

__global__
void test_shfl_swap(int * __restrict__ dOut, const int * __restrict__ dIn, int laneMask = 0x1, int firstIdx = 0, int secondIdx = 3)
{
    int idx = static_cast<int>(threadIdx.x) * kSegment;
    int value[kSegment];

    for (int i = 0; i < kSegment; i++) value[i] = dIn[idx + i];
    swap(value, static_cast<int>(threadIdx.x), laneMask, firstIdx, secondIdx);
    for (int i = 0; i < kSegment; i++) dOut[idx + i] = value[i];
}

```
| init | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|:--:|
| shfl | 7 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 8 | 9 | 10 | 15 | 12 | 13 | 14 | 11 |

#### 📌 Parallel Reduction Using the Warp Shuffle Instruction

```c++
__global__ void reduceSmemUnrollShfl(int * g_idata, int * g_odata, unsigned int n)
{
    // static shared memory
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // global index
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling 4 blocks
    int localSum = 0;

    if (idx + 3 * blockDim.x < n)
    {
        float a1 = g_idata[idx];
        float a2 = g_idata[idx + blockDim.x];
        float a3 = g_idata[idx + 2 * blockDim.x];
        float a4 = g_idata[idx + 3 * blockDim.x];
        localSum = a1 + a2 + a3 + a4;
    }

    smem[tid] = localSum;
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >=  512 && tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >=  256 && tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >=  128 && tid <  64) smem[tid] += smem[tid +  64];
    __syncthreads();
    if (blockDim.x >=   64 && tid <  32) smem[tid] += smem[tid +  32];
    __syncthreads();

    // unrolling warp
    localSum = smem[tid];

    if (tid < 32)
    {
        localSum += __shfl_xor(localSum, 16);
        localSum += __shfl_xor(localSum,  8);
        localSum += __shfl_xor(localSum,  4);
        localSum += __shfl_xor(localSum,  2);
        localSum += __shfl_xor(localSum,  1);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = localSum;
}
```



### 🎯 

#### 📌 















## 🌱 

### 🎯 

#### 📌 

## 🌱 

### 🎯 

#### 📌 
