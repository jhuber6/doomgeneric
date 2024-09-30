# amdgpu-loader

A modified version of the loader utility used for testing in the LLVM C library.
This uses the [HSA implementation](https://github.com/ROCm/ROCR-Runtime) to
interface with the GPU. We also depend on the RPC server implementation from the
LLVM C library as well as the LLVM libraries themselves. The Makefile is
unlikely to work unmodified on your system.

It initializes the RPC interface, launches the `_start` kernel, and then listens
on the RPC server to handle any requests the GPU makes. Because this runs DOOM,
it also initializes an SDL2 window. The loader then exposes two function
pointers to draw the screen and read input. The RPC implementation allows us to
then call these functions. So, the portions that run on the CPU are to load the
GPU program, handle file I/O, write the framebuffer to the screen, and read the
input queue. Everything else is done on the GPU.
