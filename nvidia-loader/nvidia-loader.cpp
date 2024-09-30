//===-- Main entry into the loader interface ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file opens a device image passed on the command line and passes it to
// one of the loader implementations for launch.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/file.h>
#include <tuple>
#include <utility>

#include "cuda.h"

#include <llvmlibc_rpc_opcodes.h>
#include <llvmlibc_rpc_server.h>

#include <SDL2/SDL.h>

#include "doomgeneric.h"
#include "doomkeys.h"

using namespace llvm;
using namespace object;

static cl::OptionCategory loader_category("loader options");

static cl::opt<bool> help("h", cl::desc("Alias for -help"), cl::Hidden,
                          cl::cat(loader_category));

static cl::opt<unsigned>
    threads_x("threads-x", cl::desc("Number of threads in the 'x' dimension"),
              cl::init(1), cl::cat(loader_category));
static cl::opt<unsigned>
    threads_y("threads-y", cl::desc("Number of threads in the 'y' dimension"),
              cl::init(1), cl::cat(loader_category));
static cl::opt<unsigned>
    threads_z("threads-z", cl::desc("Number of threads in the 'z' dimension"),
              cl::init(1), cl::cat(loader_category));
static cl::alias threads("threads", cl::aliasopt(threads_x),
                         cl::desc("Alias for --threads-x"),
                         cl::cat(loader_category));

static cl::opt<unsigned>
    blocks_x("blocks-x", cl::desc("Number of blocks in the 'x' dimension"),
             cl::init(1), cl::cat(loader_category));
static cl::opt<unsigned>
    blocks_y("blocks-y", cl::desc("Number of blocks in the 'y' dimension"),
             cl::init(1), cl::cat(loader_category));
static cl::opt<unsigned>
    blocks_z("blocks-z", cl::desc("Number of blocks in the 'z' dimension"),
             cl::init(1), cl::cat(loader_category));
static cl::alias blocks("blocks", cl::aliasopt(blocks_x),
                        cl::desc("Alias for --blocks-x"),
                        cl::cat(loader_category));

static cl::opt<bool>
    print_resource_usage("print-resource-usage",
                         cl::desc("Output resource usage of launched kernels"),
                         cl::init(false), cl::cat(loader_category));

static cl::opt<bool>
    no_parallelism("no-parallelism",
                   cl::desc("Allows only a single process to use the GPU at a "
                            "time. Useful to suppress out-of-resource errors"),
                   cl::init(false), cl::cat(loader_category));

static cl::opt<std::string> file(cl::Positional, cl::Required,
                                 cl::desc("<gpu executable>"),
                                 cl::cat(loader_category));
static cl::list<std::string> args(cl::ConsumeAfter,
                                  cl::desc("<program arguments>..."),
                                  cl::cat(loader_category));

/// Generic launch parameters for configuration the number of blocks / threads.
struct LaunchParameters {
  uint32_t num_threads_x;
  uint32_t num_threads_y;
  uint32_t num_threads_z;
  uint32_t num_blocks_x;
  uint32_t num_blocks_y;
  uint32_t num_blocks_z;
};

/// The arguments to the '_begin' kernel.
struct begin_args_t {
  int argc;
  void *argv;
  void *envp;
};

/// The arguments to the '_start' kernel.
struct start_args_t {
  int argc;
  void *argv;
  void *envp;
  void *ret;
};

/// The arguments to the '_end' kernel.
struct end_args_t {
  int argc;
};

/// Return \p V aligned "upwards" according to \p Align.
template <typename V, typename A> inline V align_up(V val, A align) {
  return ((val + V(align) - 1) / V(align)) * V(align);
}

/// Copy the system's argument vector to GPU memory allocated using \p alloc.
template <typename Allocator>
void *copy_argument_vector(int argc, const char **argv, Allocator alloc) {
  size_t argv_size = sizeof(char *) * (argc + 1);
  size_t str_size = 0;
  for (int i = 0; i < argc; ++i)
    str_size += strlen(argv[i]) + 1;

  // We allocate enough space for a null terminated array and all the strings.
  void *dev_argv = alloc(argv_size + str_size);
  if (!dev_argv)
    return nullptr;

  // Store the strings linerally in the same memory buffer.
  void *dev_str = reinterpret_cast<uint8_t *>(dev_argv) + argv_size;
  for (int i = 0; i < argc; ++i) {
    size_t size = strlen(argv[i]) + 1;
    std::memcpy(dev_str, argv[i], size);
    static_cast<void **>(dev_argv)[i] = dev_str;
    dev_str = reinterpret_cast<uint8_t *>(dev_str) + size;
  }

  // Ensure the vector is null terminated.
  reinterpret_cast<void **>(dev_argv)[argc] = nullptr;
  return dev_argv;
}

/// Copy the system's environment to GPU memory allocated using \p alloc.
template <typename Allocator>
void *copy_environment(const char **envp, Allocator alloc) {
  int envc = 0;
  for (const char **env = envp; *env != 0; ++env)
    ++envc;

  return copy_argument_vector(envc, envp, alloc);
}

inline void handle_error_impl(const char *file, int32_t line, const char *msg) {
  fprintf(stderr, "%s:%d:0: Error: %s\n", file, line, msg);
  exit(EXIT_FAILURE);
}

inline void handle_error_impl(const char *file, int32_t line,
                              rpc_status_t err) {
  fprintf(stderr, "%s:%d:0: Error: %d\n", file, line, err);
  exit(EXIT_FAILURE);
}
#define handle_error(X) handle_error_impl(__FILE__, __LINE__, X)

[[noreturn]] void report_error(Error E) {
  outs().flush();
  logAllUnhandledErrors(std::move(E), WithColor::error(errs(), "loader"));
  exit(EXIT_FAILURE);
}

std::string get_main_executable(const char *name) {
  void *ptr = (void *)(intptr_t)&get_main_executable;
  auto cow_path = sys::fs::getMainExecutable(name, ptr);
  return sys::path::parent_path(cow_path).str();
}

static void handle_error_impl(const char *file, int32_t line, CUresult err) {
  if (err == CUDA_SUCCESS)
    return;

  const char *err_str = nullptr;
  CUresult result = cuGetErrorString(err, &err_str);
  if (result != CUDA_SUCCESS)
    fprintf(stderr, "%s:%d:0: Unknown Error\n", file, line);
  else
    fprintf(stderr, "%s:%d:0: Error: %s\n", file, line, err_str);
  exit(1);
}

// Gets the names of all the globals that contain functions to initialize or
// deinitialize. We need to do this manually because the NVPTX toolchain does
// not contain the necessary binary manipulation tools.
template <typename Alloc>
Expected<void *> get_ctor_dtor_array(const void *image, const size_t size,
                                     Alloc allocator, CUmodule binary) {
  auto mem_buffer = MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char *>(image), size), "image",
      /*RequiresNullTerminator=*/false);
  Expected<ELF64LEObjectFile> elf_or_err =
      ELF64LEObjectFile::create(*mem_buffer);
  if (!elf_or_err)
    handle_error(toString(elf_or_err.takeError()).c_str());

  std::vector<std::pair<const char *, uint16_t>> ctors;
  std::vector<std::pair<const char *, uint16_t>> dtors;
  // CUDA has no way to iterate over all the symbols so we need to inspect the
  // ELF directly using the LLVM libraries.
  for (const auto &symbol : elf_or_err->symbols()) {
    auto name_or_err = symbol.getName();
    if (!name_or_err)
      handle_error(toString(name_or_err.takeError()).c_str());

    // Search for all symbols that contain a constructor or destructor.
    if (!name_or_err->starts_with("__init_array_object_") &&
        !name_or_err->starts_with("__fini_array_object_"))
      continue;

    uint16_t priority;
    if (name_or_err->rsplit('_').second.getAsInteger(10, priority))
      handle_error("Invalid priority for constructor or destructor");

    if (name_or_err->starts_with("__init"))
      ctors.emplace_back(std::make_pair(name_or_err->data(), priority));
    else
      dtors.emplace_back(std::make_pair(name_or_err->data(), priority));
  }
  // Lower priority constructors are run before higher ones. The reverse is true
  // for destructors.
  llvm::sort(ctors, [](auto x, auto y) { return x.second < y.second; });
  llvm::sort(dtors, [](auto x, auto y) { return x.second < y.second; });

  // Allocate host pinned memory to make these arrays visible to the GPU.
  CUdeviceptr *dev_memory = reinterpret_cast<CUdeviceptr *>(allocator(
      ctors.size() * sizeof(CUdeviceptr) + dtors.size() * sizeof(CUdeviceptr)));
  uint64_t global_size = 0;

  // Get the address of the global and then store the address of the constructor
  // function to call in the constructor array.
  CUdeviceptr *dev_ctors_start = dev_memory;
  CUdeviceptr *dev_ctors_end = dev_ctors_start + ctors.size();
  for (uint64_t i = 0; i < ctors.size(); ++i) {
    CUdeviceptr dev_ptr;
    if (CUresult err =
            cuModuleGetGlobal(&dev_ptr, &global_size, binary, ctors[i].first))
      handle_error(err);
    if (CUresult err =
            cuMemcpyDtoH(&dev_ctors_start[i], dev_ptr, sizeof(uintptr_t)))
      handle_error(err);
  }

  // Get the address of the global and then store the address of the destructor
  // function to call in the destructor array.
  CUdeviceptr *dev_dtors_start = dev_ctors_end;
  CUdeviceptr *dev_dtors_end = dev_dtors_start + dtors.size();
  for (uint64_t i = 0; i < dtors.size(); ++i) {
    CUdeviceptr dev_ptr;
    if (CUresult err =
            cuModuleGetGlobal(&dev_ptr, &global_size, binary, dtors[i].first))
      handle_error(err);
    if (CUresult err =
            cuMemcpyDtoH(&dev_dtors_start[i], dev_ptr, sizeof(uintptr_t)))
      handle_error(err);
  }

  // Obtain the address of the pointers the startup implementation uses to
  // iterate the constructors and destructors.
  CUdeviceptr init_start;
  if (CUresult err = cuModuleGetGlobal(&init_start, &global_size, binary,
                                       "__init_array_start"))
    handle_error(err);
  CUdeviceptr init_end;
  if (CUresult err = cuModuleGetGlobal(&init_end, &global_size, binary,
                                       "__init_array_end"))
    handle_error(err);
  CUdeviceptr fini_start;
  if (CUresult err = cuModuleGetGlobal(&fini_start, &global_size, binary,
                                       "__fini_array_start"))
    handle_error(err);
  CUdeviceptr fini_end;
  if (CUresult err = cuModuleGetGlobal(&fini_end, &global_size, binary,
                                       "__fini_array_end"))
    handle_error(err);

  // Copy the pointers to the newly written array to the symbols so the startup
  // implementation can iterate them.
  if (CUresult err =
          cuMemcpyHtoD(init_start, &dev_ctors_start, sizeof(uintptr_t)))
    handle_error(err);
  if (CUresult err = cuMemcpyHtoD(init_end, &dev_ctors_end, sizeof(uintptr_t)))
    handle_error(err);
  if (CUresult err =
          cuMemcpyHtoD(fini_start, &dev_dtors_start, sizeof(uintptr_t)))
    handle_error(err);
  if (CUresult err = cuMemcpyHtoD(fini_end, &dev_dtors_end, sizeof(uintptr_t)))
    handle_error(err);

  return dev_memory;
}

void print_kernel_resources(CUmodule binary, const char *kernel_name) {
  CUfunction function;
  if (CUresult err = cuModuleGetFunction(&function, binary, kernel_name))
    handle_error(err);
  int num_regs;
  if (CUresult err =
          cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    handle_error(err);
  printf("Executing kernel %s:\n", kernel_name);
  printf("%6s registers: %d\n", kernel_name, num_regs);
}

template <typename args_t>
CUresult launch_kernel(CUmodule binary, CUstream stream,
                       rpc_device_t rpc_device, const LaunchParameters &params,
                       const char *kernel_name, args_t kernel_args,
                       bool print_resource_usage) {
  // look up the '_start' kernel in the loaded module.
  CUfunction function;
  if (CUresult err = cuModuleGetFunction(&function, binary, kernel_name))
    handle_error(err);

  // Set up the arguments to the '_start' kernel on the GPU.
  uint64_t args_size = sizeof(args_t);
  void *args_config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &kernel_args,
                         CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                         CU_LAUNCH_PARAM_END};

  // Initialize a non-blocking CUDA stream to allocate memory if needed. This
  // needs to be done on a separate stream or else it will deadlock with the
  // executing kernel.
  CUstream memory_stream;
  if (CUresult err = cuStreamCreate(&memory_stream, CU_STREAM_NON_BLOCKING))
    handle_error(err);

  rpc_register_callback(
      rpc_device, RPC_MALLOC,
      [](rpc_port_t port, void *data) {
        auto malloc_handler = [](rpc_buffer_t *buffer, void *data) -> void {
          uint64_t size = buffer->data[0];
          void *dev_ptr;
          if (CUresult err = cuMemAllocHost(&dev_ptr, size))
            handle_error(err);
          buffer->data[0] = reinterpret_cast<uintptr_t>(dev_ptr);
        };
        rpc_recv_and_send(port, malloc_handler, data);
      },
      &memory_stream);
  rpc_register_callback(
      rpc_device, RPC_FREE,
      [](rpc_port_t port, void *data) {
        auto free_handler = [](rpc_buffer_t *buffer, void *data) {
          if (CUresult err =
                  cuMemFreeHost(reinterpret_cast<void *>(buffer->data[0])))
            handle_error(err);
        };
        rpc_recv_and_send(port, free_handler, data);
      },
      &memory_stream);

  if (print_resource_usage)
    print_kernel_resources(binary, kernel_name);

  // Call the kernel with the given arguments.
  if (CUresult err = cuLaunchKernel(
          function, params.num_blocks_x, params.num_blocks_y,
          params.num_blocks_z, params.num_threads_x, params.num_threads_y,
          params.num_threads_z, 0, stream, nullptr, args_config))
    handle_error(err);

  // Wait until the kernel has completed execution on the device. Periodically
  // check the RPC client for work to be performed on the server.
  // FIXME: This isn't legal with blocking kernels, need a separate thread.
  while (cuStreamQuery(stream) == CUDA_ERROR_NOT_READY)
    if (rpc_status_t err = rpc_handle_server(rpc_device))
      handle_error(err);

  // Handle the server one more time in case the kernel exited with a pending
  // send still in flight.
  if (rpc_status_t err = rpc_handle_server(rpc_device))
    handle_error(err);

  return CUDA_SUCCESS;
}

void *screen_buffer;

static SDL_Window *window = nullptr;
static SDL_Renderer *renderer = nullptr;
static SDL_Texture *texture = nullptr;

static unsigned char convertToDoomKey(unsigned int key) {
  switch (key) {
  case SDLK_RETURN:
    key = KEY_ENTER;
    break;
  case SDLK_ESCAPE:
    key = KEY_ESCAPE;
    break;
  case SDLK_LEFT:
    key = KEY_LEFTARROW;
    break;
  case SDLK_RIGHT:
    key = KEY_RIGHTARROW;
    break;
  case SDLK_UP:
    key = KEY_UPARROW;
    break;
  case SDLK_DOWN:
    key = KEY_DOWNARROW;
    break;
  case SDLK_LCTRL:
  case SDLK_RCTRL:
    key = KEY_FIRE;
    break;
  case SDLK_SPACE:
    key = KEY_USE;
    break;
  case SDLK_LSHIFT:
  case SDLK_RSHIFT:
    key = KEY_RSHIFT;
    break;
  case SDLK_LALT:
  case SDLK_RALT:
    key = KEY_LALT;
    break;
  case SDLK_F2:
    key = KEY_F2;
    break;
  case SDLK_F3:
    key = KEY_F3;
    break;
  case SDLK_F4:
    key = KEY_F4;
    break;
  case SDLK_F5:
    key = KEY_F5;
    break;
  case SDLK_F6:
    key = KEY_F6;
    break;
  case SDLK_F7:
    key = KEY_F7;
    break;
  case SDLK_F8:
    key = KEY_F8;
    break;
  case SDLK_F9:
    key = KEY_F9;
    break;
  case SDLK_F10:
    key = KEY_F10;
    break;
  case SDLK_F11:
    key = KEY_F11;
    break;
  case SDLK_EQUALS:
  case SDLK_PLUS:
    key = KEY_EQUALS;
    break;
  case SDLK_MINUS:
    key = KEY_MINUS;
    break;
  default:
    key = tolower(key);
    break;
  }

  return key;
}

static void init_sdl_windows() {
  if (SDL_Init(SDL_INIT_VIDEO))
    handle_error(SDL_GetError());

  window =
      SDL_CreateWindow("DOOM", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                       DOOMGENERIC_RESX, DOOMGENERIC_RESY, SDL_WINDOW_SHOWN);

  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  SDL_RenderClear(renderer);
  SDL_RenderPresent(renderer);

  texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888,
                              SDL_TEXTUREACCESS_TARGET, DOOMGENERIC_RESX,
                              DOOMGENERIC_RESY);
}

#define KEYQUEUE_SIZE 16

static unsigned short s_KeyQueue[KEYQUEUE_SIZE];
static unsigned int s_KeyQueueWriteIndex = 0;
static unsigned int s_KeyQueueReadIndex = 0;

static void addKeyToQueue(int pressed, unsigned int keyCode) {
  unsigned char key = convertToDoomKey(keyCode);

  unsigned short keyData = (pressed << 8) | key;

  s_KeyQueue[s_KeyQueueWriteIndex] = keyData;
  s_KeyQueueWriteIndex++;
  s_KeyQueueWriteIndex %= KEYQUEUE_SIZE;
}

// Function pointer the RPC implementation will call.
static void sdl_get_input(void *args) {
  uint32_t *key = *reinterpret_cast<uint32_t **>(args);

  if (s_KeyQueueReadIndex == s_KeyQueueWriteIndex) {
    *key = 0;
  } else {
    *key = s_KeyQueue[s_KeyQueueReadIndex];
    s_KeyQueueReadIndex++;
    s_KeyQueueReadIndex %= KEYQUEUE_SIZE;
  }
}

// Function pointer the RPC implementation will call.
static void sdl_draw(void *args) {
  void *buffer = *reinterpret_cast<void **>(args);

  SDL_UpdateTexture(texture, NULL, buffer, DOOMGENERIC_RESX * sizeof(uint32_t));

  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, NULL, NULL);
  SDL_RenderPresent(renderer);

  // Poll the events.
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    if (e.type == SDL_QUIT) {
      puts("Quit requested");
      atexit(SDL_Quit);
      exit(1);
    }
    if (e.type == SDL_KEYDOWN)
      addKeyToQueue(1, e.key.keysym.sym);
    else if (e.type == SDL_KEYUP)
      addKeyToQueue(0, e.key.keysym.sym);
  }
}

int load(int argc, const char **argv, const char **envp, void *image,
         size_t size, const LaunchParameters &params,
         bool print_resource_usage) {
  if (CUresult err = cuInit(0))
    handle_error(err);
  // Obtain the first device found on the system.
  uint32_t device_id = 0;
  CUdevice device;
  if (CUresult err = cuDeviceGet(&device, device_id))
    handle_error(err);

  // Initialize the CUDA context and claim it for this execution.
  CUcontext context;
  if (CUresult err = cuDevicePrimaryCtxRetain(&context, device))
    handle_error(err);
  if (CUresult err = cuCtxSetCurrent(context))
    handle_error(err);

  // Increase the stack size per thread.
  // TODO: We should allow this to be passed in so only the tests that require a
  // larger stack can specify it to save on memory usage.
  if (CUresult err = cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 3 * 1024))
    handle_error(err);

  // Initialize a non-blocking CUDA stream to execute the kernel.
  CUstream stream;
  if (CUresult err = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING))
    handle_error(err);

  // Load the image into a CUDA module.
  CUmodule binary;
  if (CUresult err = cuModuleLoadDataEx(&binary, image, 0, nullptr, nullptr))
    handle_error(err);

  // Allocate pinned memory on the host to hold the pointer array for the
  // copied argv and allow the GPU device to access it.
  auto allocator = [&](uint64_t size) -> void * {
    void *dev_ptr;
    if (CUresult err = cuMemAllocHost(&dev_ptr, size))
      handle_error(err);
    return dev_ptr;
  };

  auto memory_or_err = get_ctor_dtor_array(image, size, allocator, binary);
  if (!memory_or_err)
    handle_error(toString(memory_or_err.takeError()).c_str());

  void *dev_argv = copy_argument_vector(argc, argv, allocator);
  if (!dev_argv)
    handle_error("Failed to allocate device argv");

  // Allocate pinned memory on the host to hold the pointer array for the
  // copied environment array and allow the GPU device to access it.
  void *dev_envp = copy_environment(envp, allocator);
  if (!dev_envp)
    handle_error("Failed to allocate device environment");

  // Allocate space for the return pointer and initialize it to zero.
  CUdeviceptr dev_ret;
  if (CUresult err = cuMemAlloc(&dev_ret, sizeof(int)))
    handle_error(err);
  if (CUresult err = cuMemsetD32(dev_ret, 0, 1))
    handle_error(err);

  void *key_buffer;
  if (CUresult err = cuMemAllocHost(&key_buffer, sizeof(uint32_t *)))
    handle_error(err);
  void **storage;

  if (CUresult err =
          cuMemAllocHost(&screen_buffer, DOOMGENERIC_RESX * DOOMGENERIC_RESY *
                                             sizeof(uint32_t)))
    handle_error(err);

  std::pair<const char *, void *> symbols[] = {
      {"key_buffer", reinterpret_cast<void *>(key_buffer)},
      {"draw_framebuffer", reinterpret_cast<void *>(sdl_draw)},
      {"get_input", reinterpret_cast<void *>(sdl_get_input)}};
  for (auto &[string, value] : symbols) {
    CUdeviceptr addr = 0;
    uint64_t size = sizeof(void *);
    if (CUresult err = cuModuleGetGlobal(&addr, &size, binary, string))
      handle_error(err);

    if (CUresult err = cuMemcpyHtoD(addr, &value, sizeof(void *)))
      handle_error(err);
  }

  uint32_t warp_size = 32;
  auto rpc_alloc = [](uint64_t size, void *) -> void * {
    void *dev_ptr;
    if (CUresult err = cuMemAllocHost(&dev_ptr, size))
      handle_error(err);
    return dev_ptr;
  };
  rpc_device_t rpc_device;
  if (rpc_status_t err = rpc_server_init(&rpc_device, RPC_MAXIMUM_PORT_COUNT,
                                         warp_size, rpc_alloc, nullptr))
    handle_error(err);

  // Initialize the RPC client on the device by copying the local data to the
  // device's internal pointer.
  CUdeviceptr rpc_client_dev = 0;
  uint64_t client_ptr_size = sizeof(void *);
  if (CUresult err = cuModuleGetGlobal(&rpc_client_dev, &client_ptr_size,
                                       binary, rpc_client_symbol_name))
    handle_error(err);

  CUdeviceptr rpc_client_host = 0;
  if (CUresult err =
          cuMemcpyDtoH(&rpc_client_host, rpc_client_dev, sizeof(void *)))
    handle_error(err);
  if (CUresult err =
          cuMemcpyHtoD(rpc_client_host, rpc_get_client_buffer(rpc_device),
                       rpc_get_client_size()))
    handle_error(err);

  LaunchParameters single_threaded_params = {1, 1, 1, 1, 1, 1};
  begin_args_t init_args = {argc, dev_argv, dev_envp};
  if (CUresult err =
          launch_kernel(binary, stream, rpc_device, single_threaded_params,
                        "_begin", init_args, print_resource_usage))
    handle_error(err);

  start_args_t args = {argc, dev_argv, dev_envp,
                       reinterpret_cast<void *>(dev_ret)};
  if (CUresult err = launch_kernel(binary, stream, rpc_device, params, "_start",
                                   args, print_resource_usage))
    handle_error(err);

  // Copy the return value back from the kernel and wait.
  int host_ret = 0;
  if (CUresult err = cuMemcpyDtoH(&host_ret, dev_ret, sizeof(int)))
    handle_error(err);

  if (CUresult err = cuStreamSynchronize(stream))
    handle_error(err);

  end_args_t fini_args = {host_ret};
  if (CUresult err =
          launch_kernel(binary, stream, rpc_device, single_threaded_params,
                        "_end", fini_args, print_resource_usage))
    handle_error(err);

  // Free the memory allocated for the device.
  if (CUresult err = cuMemFreeHost(*memory_or_err))
    handle_error(err);
  if (CUresult err = cuMemFree(dev_ret))
    handle_error(err);
  if (CUresult err = cuMemFreeHost(dev_argv))
    handle_error(err);
  if (rpc_status_t err = rpc_server_shutdown(
          rpc_device, [](void *ptr, void *) { cuMemFreeHost(ptr); }, nullptr))
    handle_error(err);

  // Destroy the context and the loaded binary.
  if (CUresult err = cuModuleUnload(binary))
    handle_error(err);
  if (CUresult err = cuDevicePrimaryCtxRelease(device))
    handle_error(err);
  return host_ret;
}

int main(int argc, const char **argv, const char **envp) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  cl::HideUnrelatedOptions(loader_category);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A utility used to launch unit tests built for a GPU target. This is\n"
      "intended to provide an intrface simular to cross-compiling emulators\n");

  if (help) {
    cl::PrintHelpMessage();
    return EXIT_SUCCESS;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> image_or_err =
      MemoryBuffer::getFileOrSTDIN(file);
  if (std::error_code ec = image_or_err.getError())
    report_error(errorCodeToError(ec));
  MemoryBufferRef image = **image_or_err;

  SmallVector<const char *> new_argv = {file.c_str()};
  llvm::transform(args, std::back_inserter(new_argv),
                  [](const std::string &arg) { return arg.c_str(); });

  // Claim a file lock on the executable so only a single process can enter this
  // region if requested. This prevents the loader from spurious failures.
  int fd = -1;
  if (no_parallelism) {
    fd = open(get_main_executable(argv[0]).c_str(), O_RDONLY);
    if (flock(fd, LOCK_EX) == -1)
      report_error(createStringError("Failed to lock '%s': %s", argv[0],
                                     strerror(errno)));
  }

  init_sdl_windows();

  // Drop the loader from the program arguments.
  LaunchParameters params{threads_x, threads_y, threads_z,
                          blocks_x,  blocks_y,  blocks_z};
  int ret = load(new_argv.size(), new_argv.data(), envp,
                 const_cast<char *>(image.getBufferStart()),
                 image.getBufferSize(), params, print_resource_usage);

  if (no_parallelism) {
    if (flock(fd, LOCK_UN) == -1)
      report_error(createStringError("Failed to unlock '%s': %s", argv[0],
                                     strerror(errno)));
  }

  return ret;
}
