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

#if defined(__has_include)
#if __has_include("hsa/hsa.h")
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#elif __has_include("hsa.h")
#include "hsa.h"
#include "hsa_ext_amd.h"
#endif
#else
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#endif

#include <llvmlibc_rpc_opcodes.h>
#include <llvmlibc_rpc_server.h>

#include <SDL2/SDL.h>

#include "doomgeneric.h"
#include "doomkeys.h"

using namespace llvm;

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

// The implicit arguments of COV5 AMDGPU kernels.
struct implicit_args_t {
  uint32_t grid_size_x;
  uint32_t grid_size_y;
  uint32_t grid_size_z;
  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint8_t Unused0[46];
  uint16_t grid_dims;
  uint8_t Unused1[190];
};

/// Print the error code and exit if \p code indicates an error.
static void handle_error_impl(const char *file, int32_t line,
                              hsa_status_t code) {
  if (code == HSA_STATUS_SUCCESS || code == HSA_STATUS_INFO_BREAK)
    return;

  const char *desc;
  if (hsa_status_string(code, &desc) != HSA_STATUS_SUCCESS)
    desc = "Unknown error";
  fprintf(stderr, "%s:%d:0: Error: %s\n", file, line, desc);
  exit(EXIT_FAILURE);
}

/// Generic interface for iterating using the HSA callbacks.
template <typename elem_ty, typename func_ty, typename callback_ty>
hsa_status_t iterate(func_ty func, callback_ty cb) {
  auto l = [](elem_ty elem, void *data) -> hsa_status_t {
    callback_ty *unwrapped = static_cast<callback_ty *>(data);
    return (*unwrapped)(elem);
  };
  return func(l, static_cast<void *>(&cb));
}

/// Generic interface for iterating using the HSA callbacks.
template <typename elem_ty, typename func_ty, typename func_arg_ty,
          typename callback_ty>
hsa_status_t iterate(func_ty func, func_arg_ty func_arg, callback_ty cb) {
  auto l = [](elem_ty elem, void *data) -> hsa_status_t {
    callback_ty *unwrapped = static_cast<callback_ty *>(data);
    return (*unwrapped)(elem);
  };
  return func(func_arg, l, static_cast<void *>(&cb));
}

/// Iterate through all availible agents.
template <typename callback_ty>
hsa_status_t iterate_agents(callback_ty callback) {
  return iterate<hsa_agent_t>(hsa_iterate_agents, callback);
}

/// Iterate through all availible memory pools.
template <typename callback_ty>
hsa_status_t iterate_agent_memory_pools(hsa_agent_t agent, callback_ty cb) {
  return iterate<hsa_amd_memory_pool_t>(hsa_amd_agent_iterate_memory_pools,
                                        agent, cb);
}

template <hsa_device_type_t flag>
hsa_status_t get_agent(hsa_agent_t *output_agent) {
  // Find the first agent with a matching device type.
  auto cb = [&](hsa_agent_t hsa_agent) -> hsa_status_t {
    hsa_device_type_t type;
    hsa_status_t status =
        hsa_agent_get_info(hsa_agent, HSA_AGENT_INFO_DEVICE, &type);
    if (status != HSA_STATUS_SUCCESS)
      return status;

    if (type == flag) {
      // Ensure that a GPU agent supports kernel dispatch packets.
      if (type == HSA_DEVICE_TYPE_GPU) {
        hsa_agent_feature_t features;
        status =
            hsa_agent_get_info(hsa_agent, HSA_AGENT_INFO_FEATURE, &features);
        if (status != HSA_STATUS_SUCCESS)
          return status;
        if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)
          *output_agent = hsa_agent;
      } else {
        *output_agent = hsa_agent;
      }
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  };

  return iterate_agents(cb);
}

void print_kernel_resources(const char *kernel_name) {
  fprintf(stderr, "Kernel resources on AMDGPU is not supported yet.\n");
}

/// Retrieve a global memory pool with a \p flag from the agent.
template <hsa_amd_memory_pool_global_flag_t flag>
hsa_status_t get_agent_memory_pool(hsa_agent_t agent,
                                   hsa_amd_memory_pool_t *output_pool) {
  auto cb = [&](hsa_amd_memory_pool_t memory_pool) {
    uint32_t flags;
    hsa_amd_segment_t segment;
    if (auto err = hsa_amd_memory_pool_get_info(
            memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment))
      return err;
    if (auto err = hsa_amd_memory_pool_get_info(
            memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags))
      return err;

    if (segment != HSA_AMD_SEGMENT_GLOBAL)
      return HSA_STATUS_SUCCESS;

    if (flags & flag)
      *output_pool = memory_pool;

    return HSA_STATUS_SUCCESS;
  };
  return iterate_agent_memory_pools(agent, cb);
}

template <typename args_t>
hsa_status_t launch_kernel(hsa_agent_t dev_agent, hsa_executable_t executable,
                           hsa_amd_memory_pool_t kernargs_pool,
                           hsa_amd_memory_pool_t coarsegrained_pool,
                           hsa_amd_memory_pool_t finegrained_pool,
                           hsa_queue_t *queue, rpc_device_t device,
                           const LaunchParameters &params,
                           const char *kernel_name, args_t kernel_args,
                           bool print_resource_usage) {
  // Look up the kernel in the loaded executable.
  hsa_executable_symbol_t symbol;
  if (hsa_status_t err = hsa_executable_get_symbol_by_name(
          executable, kernel_name, &dev_agent, &symbol))
    return err;

  // Register RPC callbacks for the malloc and free functions on HSA.
  auto tuple = std::make_tuple(dev_agent, coarsegrained_pool);
  rpc_register_callback(
      device, RPC_MALLOC,
      [](rpc_port_t port, void *data) {
        auto malloc_handler = [](rpc_buffer_t *buffer, void *data) -> void {
          auto &[dev_agent, pool] = *static_cast<decltype(tuple) *>(data);
          uint64_t size = buffer->data[0];
          void *dev_ptr = nullptr;
          if (hsa_status_t err =
                  hsa_amd_memory_pool_allocate(pool, size,
                                               /*flags=*/0, &dev_ptr))
            dev_ptr = nullptr;
          hsa_amd_agents_allow_access(1, &dev_agent, nullptr, dev_ptr);
          buffer->data[0] = reinterpret_cast<uintptr_t>(dev_ptr);
        };
        rpc_recv_and_send(port, malloc_handler, data);
      },
      &tuple);
  rpc_register_callback(
      device, RPC_FREE,
      [](rpc_port_t port, void *data) {
        auto free_handler = [](rpc_buffer_t *buffer, void *) {
          if (hsa_status_t err = hsa_amd_memory_pool_free(
                  reinterpret_cast<void *>(buffer->data[0])))
            handle_error(err);
        };
        rpc_recv_and_send(port, free_handler, data);
      },
      nullptr);

  // Retrieve different properties of the kernel symbol used for launch.
  uint64_t kernel;
  uint32_t args_size;
  uint32_t group_size;
  uint32_t private_size;
  bool dynamic_stack;

  std::pair<hsa_executable_symbol_info_t, void *> symbol_infos[] = {
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &args_size},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_size},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK, &dynamic_stack},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_size}};

  for (auto &[info, value] : symbol_infos)
    if (hsa_status_t err = hsa_executable_symbol_get_info(symbol, info, value))
      return err;

  // Allocate space for the kernel arguments on the host and allow the GPU agent
  // to access it.
  void *args;
  if (hsa_status_t err = hsa_amd_memory_pool_allocate(kernargs_pool, args_size,
                                                      /*flags=*/0, &args))
    handle_error(err);
  hsa_amd_agents_allow_access(1, &dev_agent, nullptr, args);

  // Initialize all the arguments (explicit and implicit) to zero, then set the
  // explicit arguments to the values created above.
  std::memset(args, 0, args_size);
  std::memcpy(args, &kernel_args, sizeof(args_t));

  // Initialize the necessary implicit arguments to the proper values.
  int dims = 1 + (params.num_blocks_y * params.num_threads_y != 1) +
             (params.num_blocks_z * params.num_threads_z != 1);
  implicit_args_t *implicit_args = reinterpret_cast<implicit_args_t *>(
      reinterpret_cast<uint8_t *>(args) + sizeof(args_t));
  implicit_args->grid_dims = dims;
  implicit_args->grid_size_x = params.num_blocks_x;
  implicit_args->grid_size_y = params.num_blocks_y;
  implicit_args->grid_size_z = params.num_blocks_z;
  implicit_args->workgroup_size_x = params.num_threads_x;
  implicit_args->workgroup_size_y = params.num_threads_y;
  implicit_args->workgroup_size_z = params.num_threads_z;

  // Obtain a packet from the queue.
  uint64_t packet_id = hsa_queue_add_write_index_relaxed(queue, 1);
  while (packet_id - hsa_queue_load_read_index_scacquire(queue) >= queue->size)
    ;

  const uint32_t mask = queue->size - 1;
  hsa_kernel_dispatch_packet_t *packet =
      static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
      (packet_id & mask);

  // Set up the packet for exeuction on the device. We currently only launch
  // with one thread on the device, forcing the rest of the wavefront to be
  // masked off.
  uint16_t setup = (dims) << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet->workgroup_size_x = params.num_threads_x;
  packet->workgroup_size_y = params.num_threads_y;
  packet->workgroup_size_z = params.num_threads_z;
  packet->reserved0 = 0;
  packet->grid_size_x = params.num_blocks_x * params.num_threads_x;
  packet->grid_size_y = params.num_blocks_y * params.num_threads_y;
  packet->grid_size_z = params.num_blocks_z * params.num_threads_z;
  packet->private_segment_size =
      dynamic_stack ? 16 * 1024 /* 16 KB */ : private_size;
  packet->group_segment_size = group_size;
  packet->kernel_object = kernel;
  packet->kernarg_address = args;
  packet->reserved2 = 0;
  // Create a signal to indicate when this packet has been completed.
  if (hsa_status_t err =
          hsa_signal_create(1, 0, nullptr, &packet->completion_signal))
    handle_error(err);

  if (print_resource_usage)
    print_kernel_resources(kernel_name);

  // Initialize the packet header and set the doorbell signal to begin execution
  // by the HSA runtime.
  uint16_t header =
      1u << HSA_PACKET_HEADER_BARRIER |
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  uint32_t header_word = header | (setup << 16u);
  __atomic_store_n((uint32_t *)&packet->header, header_word, __ATOMIC_RELEASE);
  hsa_signal_store_relaxed(queue->doorbell_signal, packet_id);

  // Wait until the kernel has completed execution on the device. Periodically
  // check the RPC client for work to be performed on the server.
  while (hsa_signal_wait_scacquire(
             packet->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0,
             /*timeout_hint=*/1024, HSA_WAIT_STATE_ACTIVE) != 0)
    if (rpc_status_t err = rpc_handle_server(device))
      handle_error(err);

  // Handle the server one more time in case the kernel exited with a pending
  // send still in flight.
  if (rpc_status_t err = rpc_handle_server(device))
    handle_error(err);

  // Destroy the resources acquired to launch the kernel and return.
  if (hsa_status_t err = hsa_amd_memory_pool_free(args))
    handle_error(err);
  if (hsa_status_t err = hsa_signal_destroy(packet->completion_signal))
    handle_error(err);

  return HSA_STATUS_SUCCESS;
}

/// Copies data from the source agent to the destination agent. The source
/// memory must first be pinned explicitly or allocated via HSA.
static hsa_status_t hsa_memcpy(void *dst, hsa_agent_t dst_agent,
                               const void *src, hsa_agent_t src_agent,
                               uint64_t size) {
  // Create a memory signal to copy information between the host and device.
  hsa_signal_t memory_signal;
  if (hsa_status_t err = hsa_signal_create(1, 0, nullptr, &memory_signal))
    return err;

  if (hsa_status_t err = hsa_amd_memory_async_copy(
          dst, dst_agent, src, src_agent, size, 0, nullptr, memory_signal))
    return err;

  while (hsa_signal_wait_scacquire(memory_signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                   UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0)
    ;

  if (hsa_status_t err = hsa_signal_destroy(memory_signal))
    return err;

  return HSA_STATUS_SUCCESS;
}

hsa_agent_t dev_agent;
hsa_agent_t host_agent;

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
  void *buffer_ptr = *reinterpret_cast<void **>(args);

  // Copy the framebuffer locally, much faster than relying on page migration.
  if (hsa_status_t err =
          hsa_memcpy(screen_buffer, host_agent, buffer_ptr, dev_agent,
                     DOOMGENERIC_RESX * DOOMGENERIC_RESY * sizeof(uint32_t)))
    handle_error(err);

  SDL_UpdateTexture(texture, NULL, screen_buffer,
                    DOOMGENERIC_RESX * sizeof(uint32_t));

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

static int load(int argc, const char **argv, const char **envp, void *image,
                size_t size, const LaunchParameters &params,
                bool print_resource_usage) {
  // Initialize the HSA runtime used to communicate with the device.
  if (hsa_status_t err = hsa_init())
    handle_error(err);

  // Register a callback when the device encounters a memory fault.
  if (hsa_status_t err = hsa_amd_register_system_event_handler(
          [](const hsa_amd_event_t *event, void *) -> hsa_status_t {
            if (event->event_type == HSA_AMD_GPU_MEMORY_FAULT_EVENT)
              return HSA_STATUS_ERROR;
            return HSA_STATUS_SUCCESS;
          },
          nullptr))
    handle_error(err);

  // Obtain a single agent for the device and host to use the HSA memory model.
  if (hsa_status_t err = get_agent<HSA_DEVICE_TYPE_GPU>(&dev_agent))
    handle_error(err);
  if (hsa_status_t err = get_agent<HSA_DEVICE_TYPE_CPU>(&host_agent))
    handle_error(err);

  // Load the code object's ISA information and executable data segments.
  hsa_code_object_t object;
  if (hsa_status_t err = hsa_code_object_deserialize(image, size, "", &object))
    handle_error(err);

  hsa_executable_t executable;
  if (hsa_status_t err = hsa_executable_create_alt(
          HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO, "",
          &executable))
    handle_error(err);

  if (hsa_status_t err =
          hsa_executable_load_code_object(executable, dev_agent, object, ""))
    handle_error(err);

  // No modifications to the executable are allowed  after this point.
  if (hsa_status_t err = hsa_executable_freeze(executable, ""))
    handle_error(err);

  // Check the validity of the loaded executable. If the agents ISA features do
  // not match the executable's code object it will fail here.
  uint32_t result;
  if (hsa_status_t err = hsa_executable_validate(executable, &result))
    handle_error(err);
  if (result)
    handle_error(HSA_STATUS_ERROR);

  // Obtain memory pools to exchange data between the host and the device. The
  // fine-grained pool acts as pinned memory on the host for DMA transfers to
  // the device, the coarse-grained pool is for allocations directly on the
  // device, and the kernerl-argument pool is for executing the kernel.
  hsa_amd_memory_pool_t kernargs_pool;
  hsa_amd_memory_pool_t finegrained_pool;
  hsa_amd_memory_pool_t coarsegrained_pool;
  if (hsa_status_t err =
          get_agent_memory_pool<HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT>(
              host_agent, &kernargs_pool))
    handle_error(err);
  if (hsa_status_t err =
          get_agent_memory_pool<HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED>(
              host_agent, &finegrained_pool))
    handle_error(err);
  if (hsa_status_t err =
          get_agent_memory_pool<HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED>(
              dev_agent, &coarsegrained_pool))
    handle_error(err);

  if (hsa_status_t err = hsa_amd_memory_pool_allocate(
          finegrained_pool,
          DOOMGENERIC_RESX * DOOMGENERIC_RESY * sizeof(uint32_t),
          /*flags=*/0, &screen_buffer))
    handle_error(err);
  hsa_amd_agents_allow_access(1, &dev_agent, nullptr, screen_buffer);

  // Allocate fine-grained memory on the host to hold the pointer array for the
  // copied argv and allow the GPU agent to access it.
  auto allocator = [&](uint64_t size) -> void * {
    void *dev_ptr = nullptr;
    if (hsa_status_t err = hsa_amd_memory_pool_allocate(finegrained_pool, size,
                                                        /*flags=*/0, &dev_ptr))
      handle_error(err);
    hsa_amd_agents_allow_access(1, &dev_agent, nullptr, dev_ptr);
    return dev_ptr;
  };
  void *dev_argv = copy_argument_vector(argc, argv, allocator);
  if (!dev_argv)
    handle_error("Failed to allocate device argv");

  // Allocate fine-grained memory on the host to hold the pointer array for the
  // copied environment array and allow the GPU agent to access it.
  void *dev_envp = copy_environment(envp, allocator);
  if (!dev_envp)
    handle_error("Failed to allocate device environment");

  // Allocate space for the return pointer and initialize it to zero.
  void *dev_ret;
  if (hsa_status_t err =
          hsa_amd_memory_pool_allocate(coarsegrained_pool, sizeof(int),
                                       /*flags=*/0, &dev_ret))
    handle_error(err);
  hsa_amd_memory_fill(dev_ret, 0, /*count=*/1);

  // Allocate finegrained memory for the RPC server and client to share.
  uint32_t wavefront_size = 0;
  if (hsa_status_t err = hsa_agent_get_info(
          dev_agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size))
    handle_error(err);

  // Set up the RPC server.
  auto tuple = std::make_tuple(dev_agent, finegrained_pool);
  auto rpc_alloc = [](uint64_t size, void *data) {
    auto &[dev_agent, finegrained_pool] = *static_cast<decltype(tuple) *>(data);
    void *dev_ptr = nullptr;
    if (hsa_status_t err = hsa_amd_memory_pool_allocate(finegrained_pool, size,
                                                        /*flags=*/0, &dev_ptr))
      handle_error(err);
    hsa_amd_agents_allow_access(1, &dev_agent, nullptr, dev_ptr);
    return dev_ptr;
  };
  rpc_device_t device;
  if (rpc_status_t err = rpc_server_init(&device, RPC_MAXIMUM_PORT_COUNT,
                                         wavefront_size, rpc_alloc, &tuple))
    handle_error(err);

  // Initialize the RPC client on the device by copying the local data to the
  // device's internal pointer.
  hsa_executable_symbol_t rpc_client_sym;
  if (hsa_status_t err = hsa_executable_get_symbol_by_name(
          executable, rpc_client_symbol_name, &dev_agent, &rpc_client_sym))
    handle_error(err);

  void *rpc_client_host;
  if (hsa_status_t err =
          hsa_amd_memory_pool_allocate(finegrained_pool, sizeof(void *),
                                       /*flags=*/0, &rpc_client_host))
    handle_error(err);
  hsa_amd_agents_allow_access(1, &dev_agent, nullptr, rpc_client_host);

  void *rpc_client_dev;
  if (hsa_status_t err = hsa_executable_symbol_get_info(
          rpc_client_sym, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
          &rpc_client_dev))
    handle_error(err);

  // Copy the address of the client buffer from the device to the host.
  if (hsa_status_t err = hsa_memcpy(rpc_client_host, host_agent, rpc_client_dev,
                                    dev_agent, sizeof(void *)))
    handle_error(err);

  void *rpc_client_buffer;
  if (hsa_status_t err =
          hsa_amd_memory_lock(const_cast<void *>(rpc_get_client_buffer(device)),
                              rpc_get_client_size(),
                              /*agents=*/nullptr, 0, &rpc_client_buffer))
    handle_error(err);

  // Copy the RPC client buffer to the address pointed to by the symbol.
  if (hsa_status_t err =
          hsa_memcpy(*reinterpret_cast<void **>(rpc_client_host), dev_agent,
                     rpc_client_buffer, host_agent, rpc_get_client_size()))
    handle_error(err);

  if (hsa_status_t err = hsa_amd_memory_unlock(
          const_cast<void *>(rpc_get_client_buffer(device))))
    handle_error(err);
  if (hsa_status_t err = hsa_amd_memory_pool_free(rpc_client_host))
    handle_error(err);

  // Obtain the GPU's fixed-frequency clock rate and copy it to the GPU.
  // If the clock_freq symbol is missing, no work to do.
  hsa_executable_symbol_t freq_sym;
  if (HSA_STATUS_SUCCESS ==
      hsa_executable_get_symbol_by_name(executable, "__llvm_libc_clock_freq",
                                        &dev_agent, &freq_sym)) {

    void *host_clock_freq;
    if (hsa_status_t err =
            hsa_amd_memory_pool_allocate(finegrained_pool, sizeof(uint64_t),
                                         /*flags=*/0, &host_clock_freq))
      handle_error(err);
    hsa_amd_agents_allow_access(1, &dev_agent, nullptr, host_clock_freq);

    if (HSA_STATUS_SUCCESS ==
        hsa_agent_get_info(dev_agent,
                           static_cast<hsa_agent_info_t>(
                               HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY),
                           host_clock_freq)) {

      void *freq_addr;
      if (hsa_status_t err = hsa_executable_symbol_get_info(
              freq_sym, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
              &freq_addr))
        handle_error(err);

      if (hsa_status_t err = hsa_memcpy(freq_addr, dev_agent, host_clock_freq,
                                        host_agent, sizeof(uint64_t)))
        handle_error(err);
    }

    if (hsa_status_t err = hsa_amd_memory_pool_free(host_clock_freq))
      handle_error(err);
  }

  void *key_buffer;
  if (hsa_status_t err =
          hsa_amd_memory_pool_allocate(finegrained_pool, sizeof(uint32_t *),
                                       /*flags=*/0, &key_buffer))
    handle_error(err);
  hsa_amd_agents_allow_access(1, &dev_agent, nullptr, key_buffer);

  // Initialize these symbols on the GPU to control interacting with the SDL
  // window.
  std::pair<const char *, void *> symbols[] = {
      {"key_buffer", reinterpret_cast<void *>(key_buffer)},
      {"draw_framebuffer", reinterpret_cast<void *>(sdl_draw)},
      {"get_input", reinterpret_cast<void *>(sdl_get_input)}};
  for (auto &[string, value] : symbols) {
    hsa_executable_symbol_t sym;
    if (hsa_status_t err = hsa_executable_get_symbol_by_name(executable, string,
                                                             &dev_agent, &sym))
      handle_error(err);

    void *storage;
    if (hsa_status_t err =
            hsa_amd_memory_pool_allocate(finegrained_pool, sizeof(void *),
                                         /*flags=*/0, &storage))
      handle_error(err);
    hsa_amd_agents_allow_access(1, &dev_agent, nullptr, storage);
    *reinterpret_cast<void **>(storage) = value;

    void *addr;
    if (hsa_status_t err = hsa_executable_symbol_get_info(
            sym, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &addr))
      handle_error(err);

    if (hsa_status_t err =
            hsa_memcpy(addr, dev_agent, storage, host_agent, sizeof(void *)))
      handle_error(err);

    if (hsa_status_t err = hsa_amd_memory_pool_free(storage))
      handle_error(err);
  }

  // Obtain a queue with the maximum (power of two) size, used to send commands
  // to the HSA runtime and launch execution on the device.
  uint64_t queue_size;
  if (hsa_status_t err = hsa_agent_get_info(
          dev_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size))
    handle_error(err);
  hsa_queue_t *queue = nullptr;
  if (hsa_status_t err =
          hsa_queue_create(dev_agent, queue_size, HSA_QUEUE_TYPE_MULTI, nullptr,
                           nullptr, UINT32_MAX, UINT32_MAX, &queue))
    handle_error(err);

  LaunchParameters single_threaded_params = {1, 1, 1, 1, 1, 1};
  begin_args_t init_args = {argc, dev_argv, dev_envp};
  if (hsa_status_t err = launch_kernel(
          dev_agent, executable, kernargs_pool, coarsegrained_pool,
          finegrained_pool, queue, device, single_threaded_params, "_begin.kd",
          init_args, print_resource_usage))
    handle_error(err);

  start_args_t args = {argc, dev_argv, dev_envp, dev_ret};
  if (hsa_status_t err =
          launch_kernel(dev_agent, executable, kernargs_pool,
                        coarsegrained_pool, finegrained_pool, queue, device,
                        params, "_start.kd", args, print_resource_usage))
    handle_error(err);

  void *host_ret;
  if (hsa_status_t err =
          hsa_amd_memory_pool_allocate(finegrained_pool, sizeof(int),
                                       /*flags=*/0, &host_ret))
    handle_error(err);
  hsa_amd_agents_allow_access(1, &dev_agent, nullptr, host_ret);

  if (hsa_status_t err =
          hsa_memcpy(host_ret, host_agent, dev_ret, dev_agent, sizeof(int)))
    handle_error(err);

  // Save the return value and perform basic clean-up.
  int ret = *static_cast<int *>(host_ret);

  end_args_t fini_args = {ret};
  if (hsa_status_t err = launch_kernel(
          dev_agent, executable, kernargs_pool, coarsegrained_pool,
          finegrained_pool, queue, device, single_threaded_params, "_end.kd",
          fini_args, print_resource_usage))
    handle_error(err);

  if (rpc_status_t err = rpc_server_shutdown(
          device, [](void *ptr, void *) { hsa_amd_memory_pool_free(ptr); },
          nullptr))
    handle_error(err);

  // Free the memory allocated for the device.
  if (hsa_status_t err = hsa_amd_memory_pool_free(dev_argv))
    handle_error(err);
  if (hsa_status_t err = hsa_amd_memory_pool_free(dev_ret))
    handle_error(err);
  if (hsa_status_t err = hsa_amd_memory_pool_free(host_ret))
    handle_error(err);

  if (hsa_status_t err = hsa_queue_destroy(queue))
    handle_error(err);

  if (hsa_status_t err = hsa_executable_destroy(executable))
    handle_error(err);

  if (hsa_status_t err = hsa_code_object_destroy(object))
    handle_error(err);

  if (hsa_status_t err = hsa_shut_down())
    handle_error(err);

  return ret;
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
