#ifndef DOOM_GENERIC
#define DOOM_GENERIC

#include <stdint.h>
#include <stdlib.h>

#ifndef DOOMGENERIC_RESX
#define DOOMGENERIC_RESX 2 * 640
#endif // DOOMGENERIC_RESX

#ifndef DOOMGENERIC_RESY
#define DOOMGENERIC_RESY 2 * 400
#endif // DOOMGENERIC_RESY

#define DOOM_DRAW_BUFFER ('d' << 24 | 0)
#define DOOM_GET_INPUT ('d' << 24 | 1)

#ifdef CMAP256

typedef uint8_t pixel_t;

#else // CMAP256

typedef uint32_t pixel_t;

#endif // CMAP256

#ifdef __cplusplus
extern "C" {
#endif

extern pixel_t *DG_ScreenBuffer;

void doomgeneric_Create(int argc, char **argv);
void doomgeneric_Tick();

// Implement below functions for your platform
void DG_Init();
void DG_DrawFrame();
void DG_SleepMs(uint32_t ms);
uint32_t DG_GetTicksMs();
int DG_GetKey(int *pressed, unsigned char *key);
void DG_SetWindowTitle(const char *title);

#ifdef __cplusplus
}
#endif

#endif // DOOM_GENERIC
