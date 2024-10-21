# C/C++ Translation And Loading Toolchain


## Translation And Loading Pipeline

![Translation And Loading Pipeline](./translation_and_loading_pipeline.png)

- Compilation
  - Done by g++
  - Four steps:
    - Preprocessing
    - Compliation Proper
    - Assembly
    - Linking (invokes GNU (static) linker ld, passes arguments to it)