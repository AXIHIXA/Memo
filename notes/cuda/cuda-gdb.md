# Notes on GDB and CUDA-GDB


## Online Documentation

- [Debugging with GDB](https://sourceware.org/gdb/current/onlinedocs/gdb)
- [CUDA-GDB](https://docs.nvidia.com/cuda/cuda-gdb/)



## [GDB](https://sourceware.org/gdb/current/onlinedocs/gdb)

### [Invoking GDB](https://sourceware.org/gdb/current/onlinedocs/gdb#Invoking-GDB)

- Invocation
  - gdb *program*: Specifying an executable program *program*
  - gdb *program core*: Start with both an executable program *program* and a core file *core* specified
  - gdb program *number*: Debug a running process, Specify a process ID *number*
  - gdb -p *number*: Debug a running process, Specify a process ID *number*
- Choosing Files
  - \-symbols *file*, or, \-s *file*: Read symbol table from file *file*
  - \-exec *file*, or, \-e *file*: Use file file as the executable *file* to execute
  - \-se *file*`: Read symbol table from file *file* and use it as the executable file
  - \-core *file*, or, \-c *file*: Use file *file* as a core dump to examine
  - \-pid *number*, or, \-p *number*: Connect to process ID number, as with the attach command.
- Command Completion
  - GDB can fill in the rest of a word in a command for you, if there is only one possibility; it can also show you what the valid possibilities are for the next word in a command, at any time. 
  - This works for GDB commands, GDB subcommands, command options, and the names of symbols in your program.
  - Press the TAB key whenever you want GDB to fill out the rest of a word

### Specifying a Debugging Target, And Start Debugging

- help *whatever*: Print help info on *whatever* stuff. 
- target exec *program*
- target core *filename*
- r (run): Run program
- a (attach): Attaches to a running process
  - attach *process\-id*
- k (kill): Kill the program being debugged
- q (quit): Exit gdb

### Debugging Programs with Multiple Threads

- info threads: Inquire about existing threads
- thread *thread\-id*: Switch among threads
- thread apply [ *thread\-id\-list* | all ] *args*: Apply a command to a list of threads

### Brief: At A Breakpoint

- n (next): Advance execution to the next line of the *current function*
  - next [ *count* ]
- s (step): Goes to the next line to be executed in *any subroutine*
  - step [ *count* ]
- c (continue): Continue executing
- p (print): Print or set a value
  - p *val*: Prints *val* to console
  - p *val*=*expr*: Sets *val* to *expr*
- l (list): Print surrounding ten lines of code of the breakpoint
  - list *linenum*: Print lines centered around line number *linenum* in the current source file.
  - list *function*: Print lines centered around the beginning of function *function*.
  - list: Print more lines

### Examining the Stack

- bt (backtrace): Display a stack frame for each active subroutine
  - bt full: Print the values of the local variables also
- f (frame): Selecting a stack frame
  - frame [ *frame\-selection\-spec* ]: Allows different stack frames to be selected
  - frame: When used without any argument, prints a brief description of the currently selected stack frame
- up *n*: Move *n* frames up the stack; n defaults to 1.
- down *n*: Move *n* frames down the stack; n defaults to 1.
- i (info): Some info commands could be used here. Not the complete set. 
  - info frame, or, info f: Prints a verbose description of the selected stack frame
  - info args: Show the arguments passed to a function
  - info locals: Print the local variables of the selected frame
  - info registers: List the registers currently in use 
  - info stack: Backtrace of the stack, or innermost COUNT frames



## [CUDA-GDB](https://docs.nvidia.com/cuda/cuda-gdb/)

### Compiling the Application

- The \-g \-G option pair must be passed to NVCC when an application is compiled for ease of debugging with CUDA-GDB
  - Forces -O0 compilation, with the exception of very limited dead-code eliminations and register-spilling optimizations.
  - Makes the compiler include debug information in the executable
- \-lineinfo can be used when trying to debug optimized code. 
  - \-G defaults \-lineinfo

### [CUDA-GDB Extensions](https://docs.nvidia.com/cuda/cuda-gdb/#cuda-gdb-extensions)

- CUDA-specific command naming convention
  - The existing GDB commands are unchanged. 
  - Every new CUDA command or option is prefixed with the CUDA keyword. 
  - E.g., 
    - GNU-GDB for host threads: `info threads`, `thread 1`
    - CUDA-GDB for CUDA threads: `info cuda threads`, `cuda thread 1`

### [GPU Core Dump Support](https://docs.nvidia.com/cuda/cuda-gdb/#gpu-core-dump-support)

- **Compilation for GPU core dump generation**
  - GPU core dumps will be generated regardless of compilation flags used to generate the GPU application
  - Recommended to compile the application with the \-g \-G or the \-lineinfo option with NVCC
- **Enabling GPU core dump generation on exception with environment variables**
  - `export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`
    - Enable generating a GPU core dump when a GPU exception is encountered. 
    - This option is disabled by default.
    - By default, a GPU core dump is created in the current working directory. 
  - Starting from CUDA 11.6, the compute-sanitizer tool can generate a GPU core dump when an error is detected by using the `--generate-coredump` yes option. Once the core dump is generated, the target application will abort. See the [compute-sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#coredump) for more information. 
- **Displaying core dump generation progress**
  - `export CUDA_COREDUMP_SHOW_PROGRESS=1`
    - Print core dump generation progress messages to stderr. 
    - Some GPU core dumps could take a long time to be generated (e.g., 30 min)
    - This can be used to determine how far along the coredump generation is. 
- **Inspecting GPU and GPU+CPU core dumps in cuda-gdb**
  - `(cuda-gdb) target cudacore core.cuda.localhost.1234`
    - Open the core dump file and print the exception encountered during program execution. 
    - Then, issue standard cuda-gdb commands to further investigate application state on the device at the moment it was aborted.
  - `(cuda-gdb) target core core.cpu core.cuda`
    - Open the core dump file and print the exception encountered during program execution. 
    - Then, issue standard cuda-gdb commands to further investigate application state on the host and the device at the moment it was aborted.
  - Coredump inspection does not require that a GPU be installed on the system

### [Kernel Focus](https://docs.nvidia.com/cuda/cuda-gdb/#kernel-focus)

#### Current Focus*
```
(cuda-gdb) cuda device sm warp lane block thread
block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0
(cuda-gdb) cuda kernel block thread
kernel 1, block (0,0,0), thread (0,0,0)
(cuda-gdb) cuda kernel
kernel 1
```
#### Switching Focus*
```
(cuda-gdb) cuda device 0 sm 1 warp 2 lane 3
[Switching focus to CUDA kernel 1, grid 2, block (8,0,0), thread
(67,0,0), device 0, sm 1, warp 2, lane 3]
374 int totalThreads = gridDim.x * blockDim.x;
```
If the specified focus is not fully defined by the command, the debugger will assume that the omitted coordinates are set to the coordinates in the current focus, including the subcoordinates of the block and thread.
```
(cuda-gdb) cuda thread (15)
[Switching focus to CUDA kernel 1, grid 2, block (8,0,0), thread
(15,0,0), device 0, sm 1, warp 0, lane 15]
374 int totalThreads = gridDim.x * blockDim.x;
```
The parentheses for the block and thread arguments are optional.
```
(cuda-gdb) cuda block 1 thread 3
[Switching focus to CUDA kernel 1, grid 2, block (1,0,0), thread (3,0,0),
device 0, sm 3, warp 0, lane 3]
374 int totalThreads = gridDim.x * blockDim.
```

### [Inspecting Program State](https://docs.nvidia.com/cuda/cuda-gdb/#inspecting-program-state)

- Depending on the variable type and usage, variables can be stored either in registers or in local, shared, const or global memory. You can print the address of any variable to find out where it is stored and directly access the associated memory.
```
(cuda-gdb) print &array
$1 = (@shared int (*)[0]) 0x20
(cuda-gdb) print array[0]@4
$2 = {0, 128, 64, 192}
```
- Access the shared memory indexed into the starting offset to see what the stored values are:
```
(cuda-gdb) print *(@shared int*)0x20
$3 = 0
(cuda-gdb) print *(@shared int*)0x24
$4 = 128
(cuda-gdb) print *(@shared int*)0x28
$5 = 64
```
- Access the starting address of the input parameter to the kernel.
```
(cuda-gdb) print &data
$6 = (const @global void * const @parameter *) 0x10
(cuda-gdb) print *(@global void * const @parameter *) 0x10
$7 = (@global void * const @parameter) 0x110000</>
```
- Info CUDA Commands
  - info cuda devices: information about all the devices
  - info cuda sms: information about all the active SMs in the current device
  - info cuda warps: information about all the active warps in the current SM
  - info cuda lanes: information about all the active lanes in the current warp
  - info cuda kernels: information about all the active kernels
  - info cuda blocks: information about all the active blocks in the current kernel
  - info cuda threads: information about all the active threads in the current kernel
