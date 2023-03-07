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
    int i = blockIdx.x * blockDim.x + threadIdx.x;The runtime creates a CUDA context for each device in the system
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
  - CUDA instruction set, called *PTX*. 

#### 📌 3.1.1. Compilation Workflow

##### 3.1.1.1. Offline Compilation

- Source files compiled with `nvcc` can include a mix of host code and device code. 
- `nvcc` separates device code from host code and then:
  - Device code: 
    - Compile device code into assembly (*PTX* code) and/or binary (*cubin* object),
  - Host code: 
    - Modify the host code by replacing the `<<<...>>>` syntax by CUDA runtime function calls 
      - which load and launch each compiled kernel from the *PTX* code and/or *cubin* object. 
    - The modified host code is output 
      - as C++ code left to another compiler, or
      - as object code directly 
        - by letting `nvcc` invoke the host compiler during the last compilation stage. 
- Applications can then
  - (the most common case) link to the compiled host code, or
  - ignore the modified host code (if any) 
    and use the CUDA driver API 
    to load and execute the *PTX* code or *cubin* object.

##### 3.1.1.2. Just-in-Time Compilation

**JIT Compilcation**. 
Any *PTX* code loaded by an application at runtime is compiled further to binary code by the device driver. 
This is called just-in-time compilation. 
Just-in-time compilation increases application load time, 
but allows the application to benefit from any new compiler improvements coming with each new device driver. 
It is also the only way for applications to run on devices that did not exist at the time the application was compiled, 
as detailed in Application Compatibility.

**Compute Cache**. 
When the device driver just-in-time compiles some *PTX* code for some application, 
it automatically caches a copy of the generated binary code 
in order to avoid repeating the compilation in subsequent invocations of the application. 
The cache (*compute cache*) is automatically invalidated when the device driver is upgraded, 
so that applications can benefit from the improvements in the new just-in-time compiler built into the device driver.

Environment variables are available to control just-in-time compilation. 

As an alternative to using `nvcc` to compile CUDA C++ device code, 
NVRTC (a runtime compilation library for CUDA C++) can be used to compile CUDA C++ device code to PTX at runtime. 

#### 📌 3.1.2. Binary Compatibility

- Binary code is architecture-specific. 
- A *cubin* object is generated using the compiler option `-code` that specifies the targeted architecture. 
  - Compiling with `-code=sm_80` produces binary code for devices of compute capability 8.0. 
- A cubin object generated for compute capability `X.y` will only execute 
  on devices of compute capability `X.z` where `z ≥ y`.

#### 📌 3.1.3. PTX Compatibility

- Some *PTX* instructions are only supported on devices of higher compute capabilities.
- The `-arch` compiler option specifies the compute capability that is assumed when compiling C++ to PTX code. 
  - E.g., *Warp Shuffle Functions* are only supported on devices of compute capability 5.0 and above. 
    - Code that contains warp shuffle must be compiled with `-arch=compute_50` (or higher).

*PTX* code produced for some specific compute capability 
can always be compiled to binary code of greater or equal compute capability. 
Note that a binary compiled from an earlier *PTX* version 
may not make use of some hardware features. 
E.g., a binary targeting devices of compute capability 7.0 (Volta) 
compiled from *PTX* generated for compute capability 6.0 (Pascal) 
will not make use of Tensor Core instructions,
since these were not available on Pascal. 
As a result, the final binary may perform worse than would be possible 
if the binary were generated using the latest version of *PTX*. 

#### 📌 3.1.4. Application Compatibility

- An application must load binary or *PTX* code 
  that is compatible with its device's compute capability. 
  - To xecute code on future architectures with higher compute capability 
    (for which no binary code can be generated yet), 
    an application must load *PTX* code that will be just-in-time compiled for these devices. 
- Which *PTX* and binary code gets embedded? Controlled by: 
  - the `-arch` and `-code` compiler options, or 
  - the `-gencode` compiler option
- E.g., the following code embeds 
  - binary code compatible with compute capability 5.0 and 6.0 (first and second `-gencode` options), and
  - *PTX* and binary code compatible with compute capability 7.0 (third `-gencode` option).
```bash
nvcc x.cu
      -gencode arch=compute_50,code=sm_50
      -gencode arch=compute_60,code=sm_60
      -gencode arch=compute_70,code=\"compute_70,sm_70\"
```
- Host code is generated to automatically select at runtime the most appropriate code to load and execute, 
  which, in the above example, will be:
  - 5.0 binary code for devices with compute capability 5.0 and 5.2,
  - 6.0 binary code for devices with compute capability 6.0 and 6.1,
  - 7.0 binary code for devices with compute capability 7.0 and 7.5,
  - *PTX* code which is compiled to binary code at runtime for devices with compute capability 8.0 and 8.6.
- The `nvcc` user manual lists various shorthands for the `-arch`, `-code`, and `-gencode` compiler options. 
  - For example, `-arch=sm_70` is a shorthand for `-arch=compute_70 -code=compute_70,sm_70` 
    (which is the same as `-gencode arch=compute_70,code=\"compute_70,sm_70\"`).

#### 📌 3.1.5. C++ Compatibility

The front end of the compiler processes CUDA source files according to C++ syntax rules. 
Full C++ is supported for the host code. 
However, only a subset of C++ is fully supported for the device code.

#### 📌 3.1.6. 64-Bit Compatibility

The 64-bit version of `nvcc` compiles device code in 64-bit mode (i.e., pointers are 64-bit). 
Device code compiled in 64-bit mode is only supported with host code compiled in 64-bit mode.

### 🎯 3.2. CUDA Runtime

#### 📌 3.2.1. Initialization

- The runtime creates a CUDA context for each device in the system. 
  - is the *primary context* for this device 
  - is initialized at the first runtime function which requires an active context on this device. 
  - is shared among all the host threads of the application.
  - This all happens transparently. 
- `cudaDeviceReset()` destroys the primary context of the host's current device.  
  - The next runtime function call made by any host thread that has this device as current 
    will create a new primary context for this device. 
- `cudaSetDevice()` will explicitly initialize the runtime after changing host's current device. 
  - Previous versions of CUDA delayed runtime initialization on the new device 
    until the first runtime call was made after `cudaSetDevice()`. 
  - This change means that it is now very important to 
    check the return value of `cudaSetDevice()` for initialization errors.
- **Never** use CUDA APIs during program initiation or after termination. 
  - The CUDA interfaces use global state that is 
    - initialized during host program initiation
    - destroyed during host program termination. 
  - The CUDA runtime and driver cannot detect if this state is invalid.  
    - Using any of these interfaces (implicitly or explicitly) 
      during program initiation or termination after main
      will result in undefined behavior.

#### 📌 3.2.2. Device Memory

- Kernels operate out of device memory. 
  - The runtime provides functions to: 
    - allocate device memory, 
    - deallocate device memory, 
    - copy device memory, 
    - transfer data between host memory and device memory. 
- Device memory can be allocated as: 
  - *Linear memory*, or 
  - CUDA arrays.
- **1D arrays**: 
  - Allocated using `cudaMalloc()`
  - Freed using `cudaFree()`
  - Data transfer between host and device done by `cudaMemcpy()`. 
```c++
// Device code
__global__ void VecAdd(float * A, float * B, float * C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)  C[i] = A[i] + B[i];
}

// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float * h_A = (float *) malloc(size);
    float * h_B = (float *) malloc(size);
    float * h_C = (float *) malloc(size);

    // Initialize input vectors
    ...

    // Allocate vectors in device memory
    float * d_A;
    cudaMalloc(&d_A, size);
    float * d_B;
    cudaMalloc(&d_B, size);
    float * d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    // a.k.a. std::ceil(static_cast<double>(N) / static_cast<double>(threadsPerBlock))
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    ...
} 
```
- **2D / 3D Arrays**: 
  - Allocated through `cudaMallocPitch()` and `cudaMalloc3D()`. 
    - guarantees that the allocation is appropriately padded 
      to meet the alignment requirements, 
    - therefore ensuring best performance when
      - accessing the row addresses, or
      - performing copies between 2D arrays and other regions of device memory 
  - Transfered using `cudaMemcpy2D()` and `cudaMemcpy3D()`. 
  - The returned pitch (or stride) must be used to access array elements. 
    - Pitch: matrix width (with padding). 
```c++
// Host code
int width = 64, height = 64;
float * devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float * devPtr, std::size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r)
    {
        auto row = reinterpret_cast<float *>(reinterpret_cast<char *>(devPtr) + r * pitch);
        
        for (int c = 0; c < width; ++c)
        {
            float element = row[c];
        }
    }
}
```
```c++
// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
    char * devPtr = devPitchedPtr.ptr;
    std::size_t pitch = devPitchedPtr.pitch;
    std::size_t slicePitch = pitch * height;
    
    for (int z = 0; z < depth; ++z)
    {
        char * slice = devPtr + z * slicePitch;
        
        for (int y = 0; y < height; ++y)
        {
            auto row = reinterpret_cast<float *>(slice + y * pitch);
            
            for (int x = 0; x < width; ++x)
            {
                float element = row[x];
            }
        }
    }
}
```
- The following code sample illustrates various ways of accessing global variables via the runtime API:
  - `cudaGetSymbolAddress()` is used to retrieve the address 
    pointing to the memory allocated for a variable declared in global memory space. 
  - The size of the allocated memory is obtained through `cudaGetSymbolSize()`.
```c++
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float * devPointer;
float * ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

#### 📌 3.2.4. Shared Memory









## 🌱 

### 🎯 

#### 📌 

## 🌱 

### 🎯 

#### 📌 
