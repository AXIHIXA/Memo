# OpenGLDemo 

## Overview

Implemented a sample program to display flat-shaded triangle, tetrahedron and sphere (with tessellation shaders). 
Also implemented a FPS-style camera and local illumination with the Phong shading model. 

## Compile & Run

Execute the following commands in the same directory of this README: 
```bash
mkdir build
cd build
cmake ..
make 
cd ..
./build/OpenGLDemo3D
```

## Usage

- Camera
  - Press `X` to show/hide the x, y, z axis. 
  - Press `W`/`S`/`A`/`D`/`UP`/`DOWN`, or drag/scroll the mouse to adjust the camera. 

## Notes

- In this program, the sphere parameters passed into tessellation shaders via shader uniforms. 
  Note how this differ from the "pass-by-vertex-attribute-array" method for the circle example; 
- If this program does not work on your VMWare virtual environment, 
  please try to [disable the 3D acceleration feature](https://kb.vmware.com/s/article/59146). 
