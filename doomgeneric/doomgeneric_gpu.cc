#include <stdio.h>
#include <time.h>

#include "doomgeneric.h"
#include "doomkeys.h"
#include "m_argv.h"

#include <gpuintrin.h>
#include <shared/rpc.h>

#define NS_IN_MS 1000000L

[[gnu::visibility("protected")]] extern rpc::Client
    client asm("__llvm_rpc_client");

void DG_Init() {}

void DG_DrawFrame() {
  auto port = client.open<DOOM_DRAW_BUFFER>();
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = reinterpret_cast<uintptr_t>(DG_ScreenBuffer);
  });
  port.close();
}

void DG_SleepMs(uint32_t ms) {
  struct timespec tim;
  tim.tv_sec = ms / 1000;
  tim.tv_nsec = (NS_IN_MS * ms) % (NS_IN_MS * 1000L);
  nanosleep(&tim, NULL);
}

uint32_t DG_GetTicksMs() {
  struct timespec tim;
  clock_gettime(CLOCK_MONOTONIC, &tim);
  return (uint32_t)(tim.tv_sec * 1000 + tim.tv_nsec / NS_IN_MS);
}

int DG_GetKey(int *pressed, unsigned char *doomKey) {
  auto port = client.open<DOOM_GET_INPUT>();
  uint32_t key = 0;
  port.send_and_recv(
      [](rpc::Buffer *, uint32_t) {},
      [&](rpc::Buffer *buffer, uint32_t) { key = buffer->data[0]; });
  port.close();
  if (key == 0)
    return 0;

  *pressed = key >> 8;
  *doomKey = key & 0xFF;

  return 1;
}

void DG_SetWindowTitle(const char *title) {}

int main(int argc, char **argv, char **envp) {
  if (__gpu_thread_id(0) == 0)
    doomgeneric_Create(argc, argv);
  __gpu_sync_threads();

#ifdef SHOWFPS
  uint32_t time = DG_GetTicksMs();
  uint32_t last_tick = 0;
#endif
  for (int i = 0;; ++i) {
    doomgeneric_Tick();

#ifdef SHOWFPS
    if (__gpu_thread_id(0) == 0) {
      int interval = 10;
      if (i % interval == 0) {
        uint32_t new_time = DG_GetTicksMs();
        uint32_t diff = (new_time - time);
        if (diff > 2000) {
          float fps = (float)(i - last_tick) / (diff / 1000.0f);
          last_tick = i;
          time = new_time;
          printf("fps %f\n", fps);
        }
      }
    }
#endif
  }

  return 0;
}
