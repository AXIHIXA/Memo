# OpenGLDemo 

## Overview

Implemented a sample program to display colored triangle (with self-spin effect)
as well as circles (with tessellation shaders). 

## Compile & Run

Execute the following commands in the same directory of this README: 
```bash
mkdir build
cd build
cmake ..
make 
cd ..
./build/OpenGLDemo
```

## Usage

- Press key R to start/stop rotation of the triangle;
- Press key C to show/hide the circles.

## Notes

- Suggested order to read and understand this program: 
  - GLFW callbacks;
  - Triangle (ignore the code related to the self-spin effect);
  - Triangle (with self-spin; involves transformation matrices);
  - Circles (involves tessellation shaders, which are not necessary in the first half of this course). 
- In this program, the circle parameters are passed into tessellation shaders via generic vertex attribute arrays. 
  Note how this differs from the "pass-by-shader-uniforms" method for the sphere example; 
- Please do remember to play with the program as guided by the comments in the tessellation evaluation shader;
- If this program does not work on your VMWare virtual environment, 
  please try to [disable the 3D acceleration feature](https://kb.vmware.com/s/article/59146). 
