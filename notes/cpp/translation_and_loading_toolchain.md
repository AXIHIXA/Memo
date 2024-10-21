# C/C++ Translation And Loading Toolchain


## Translation And Loading Pipeline

- Compilation
  - Done by g++
  - Four steps:
    - Preprocessing
    - Compliation Proper
    - Assembly
    - Linking (invokes GNU (static) linker ld, passes arguments to it)