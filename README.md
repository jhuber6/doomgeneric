# doomgpu

This is a port of DOOM that runs (almost) entirely on the GPU using the [LLVM C
library for GPUs](https://libc.llvm.org/gpu/) based on the
[doomgeneric](https://github.com/ozkl/doomgeneric) interface.

To try it you will need a WAD file (game data). If you don't own the game,
shareware version is freely available (doom1.wad).

This implementation currently only works on AMD GPUs, support for NVIDIA will
require a port of the facilities in the loader. This is also really hacky and
will likely change in the future.

# requirements

* A Linux operating system
* An AMDGPU with ROCm support
* SDL2 libraries
* A ROCm or ROCR-Runtime installation
* An LLVM build off of the main branch

# why

Because I can.

# how

The `clang` compiler can target GPUs directly. We emit a single kernel that
calls the 'main' function. Functions that require the operating system are
handled through the RPC interface. See [my LLVM
talk](https://www.youtube.com/watch?v=_LLGc48GYHc) for more information.

This implementation defines the `amdgpu-loader` utility, which handles launching
the `main` kernel, setting up the SDL2 window interface, and provides functions
to get the input keys and write the output framebuffer. Okay, it's not
*entirely* on the GPU, but all the logic and rendering runs on the GPU.

# building and running

You will need an LLVM installation with the LLVM C library for GPUs enabled.
Don't do a shared library build of LLVM it will probably break. See [the
documentation](https://libc.llvm.org/gpu/building.html#standard-runtimes-build)
for how to build it.

Once installed, use the newly built `clang` compiler to build the libraries.
Make sure that you have `include/hsa.h` and `libhsa-runtime64.so` available from
your ROCm installation.

```console
$ make -C amdgpu_loader/ -j
$ make -C doomgeneric/ -f Makefile.amdgpu -j
$ ./amdgpu-loader/amdgpu-loader --threads 512 ./doomgeneric/doomgeneric -iwad doom1.wad
```

![AMDGPU](screenshots/amdgpu.png)

Thanks to @hardcode84 for porting help.
