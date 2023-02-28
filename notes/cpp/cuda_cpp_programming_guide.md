# CUDA C++ Programming Notes

- Refer to [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html). 



## 🌱 2. Programming Model

### 🎯 2.2. Thread Hierarchy

- Kernel functions
  - One thread block ~ One Streaming Multiprocessor (SM), i.e., CUDA Core. 
  - Multiple thread blocks (organized as a grid, 1D, 2D, or 3D, `gridDim.xyz`, `blockIdx.xyz`). 
  - Inside each thread block (1D, 2D, or 3D): multiple threads (`blockDim.xyz`, `threadIdx.xyz`). 

There is a limit to the number of threads per block, 
since all threads of a block are expected to reside on the same streaming multiprocessor core 
and must share the limited memory resources of that core. 

However, a kernel can be executed by multiple equally-shaped thread blocks, 
so that the total number of threads is equal to the number of threads per block times the number of blocks. 

Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional grid of thread blocks. 
The number of thread blocks in a grid is usually dictated by the size of the data being processed, 
which typically exceeds the number of processors in the system. 

```c++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)  C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    // ...

    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);

    // ...
}
```

#### 📌 Synchronization

Threads within a block can cooperate 
by sharing data through some shared memory 
and by synchronizing their execution to coordinate memory accesses. 

More precisely, one can specify synchronization points in the kernel 
by calling the `__syncthreads()` intrinsic function, 
which acts as a barrier at which all threads in the block must wait before any is allowed to proceed. 

A rich set of thread-synchronization primitives in addition to `__syncthreads()`.
`__syncthreads()` is expected to be lightweight. 

### 🎯 2.3. Memory Hierarchy

- **Thread**: Per thread registers and local memory
- **Block**: Per block shared memory
- (**Cluster**: Shared memory of all thread blocks in a cluster)
- **Grid**: Global memory shared between all GPU kernels

### 🎯 2.4. Heterogeneous Programming

- The kernels execute on a GPU and the rest of the C++ program executes on a CPU. 
- Both the host and the device maintain their own separate memory spaces (host memory and device memory). 
  - Global, constant, and texture memory spaces. 
  - Device memory allocation & deallocation + Data transfer between host and device.
- Unified Memory provides managed memory
  - Accessible from all CPUs and GPUs in a common address space. 
  - Eliminates the need to explicitly mirror data on host and device. 



## 🌱 3. Programming Interface

- CUDA C++: 
  - **Extensions to the C++ language**
    - Allow programmers to define a kernel as a C++ function 
      and use some new syntax to specify the grid and block dimension 
      each time the function is called.
    - Any source file that contains any of these extensions 
      must be compiled with `nvcc`. 
  - **Runtime library** 
    - Provides C and C++ functions that: 
      - execute on the host to allocate and deallocate device memory, 
      - transfer data between host memory and device memory, 
      - manage systems with multiple devices, 
      - etc.
    - Built on top of a lower-level C API, the CUDA driver API. 
      - Most applications do **not** need to use the driver API. 

### 🎯 3.1. Compilation with NVCC

- Kernels must be compiled into binary code by `nvcc` to execute on the device. 
- Kernels could be written in 
  - C/C++
  - CUDA instruction set, called _PTX_. 

#### 📌 3.1.1. Compilation Workflow

- Source files compiled with `nvcc` can include a mix of host code and device code. 
- `nvcc` separates device code from host code and then:
  - Compile device code into assembly (PTX code) and/or binary (cubin object),
  - Modify the host code by replacing the `<<<...>>>` syntax by CUDA runtime function calls 
    - which load and launch each compiled kernel from the PTX code and/or cubin object. 









## 🌱 

### 🎯 

#### 📌 

## 🌱 

### 🎯 

#### 📌 
