#include <stdio.h>
#include <time.h>

#include "doomgeneric.h"
#include "doomkeys.h"
#include "gpu_utils.h"
#include "m_argv.h"

#include <gpu/rpc.h>

#define NS_IN_MS 1000000L

// Externally initialized and handled by the loader utility.
[[gnu::visibility("protected")]] void *draw_framebuffer = NULL;
[[gnu::visibility("protected")]] void *get_input = NULL;

uint32_t *key_buffer = NULL;

#define KEYQUEUE_SIZE 16

static unsigned short s_KeyQueue[KEYQUEUE_SIZE];
static unsigned int s_KeyQueueWriteIndex = 0;
static unsigned int s_KeyQueueReadIndex = 0;

static void handle_input() {
  rpc_host_call(get_input, &key_buffer, sizeof(uint32_t *));
  s_KeyQueue[s_KeyQueueWriteIndex] = *key_buffer;
  s_KeyQueueWriteIndex++;
  s_KeyQueueWriteIndex %= KEYQUEUE_SIZE;
}

void DG_Init() { key_buffer = malloc(sizeof(uint32_t)); }

void DG_DrawFrame() {
  rpc_host_call(draw_framebuffer, &DG_ScreenBuffer, sizeof(void *));
  handle_input();
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
  if (s_KeyQueueReadIndex == s_KeyQueueWriteIndex)
    return 0;

  unsigned short keyData = s_KeyQueue[s_KeyQueueReadIndex];
  s_KeyQueueReadIndex++;
  s_KeyQueueReadIndex %= KEYQUEUE_SIZE;

  *pressed = keyData >> 8;
  *doomKey = keyData & 0xFF;

  return 1;
}

void DG_SetWindowTitle(const char *title) {}

int main(int argc, char **argv, char **envp) {
  uint32_t thread_id = 0;
  if (thread_id == 0) {
    doomgeneric_Create(argc, argv);

    uint32_t time = DG_GetTicksMs();
    uint32_t last_tick = 0;
    for (int i = 0;; ++i) {

      doomgeneric_Tick();
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
  }

  return 0;
}
