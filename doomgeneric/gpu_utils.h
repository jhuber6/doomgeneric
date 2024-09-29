#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <stdint.h>

// Type aliases to the address spaces used by the AMDGPU backend.
#define Private [[clang::opencl_private]]
#define Constant [[clang::opencl_constant]]
#define Local [[clang::opencl_local]]
#define Global [[clang::opencl_global]]

// Returns the number of workgroups in the 'x' dimension of the grid.
static inline uint32_t get_num_blocks_x() {
  return __builtin_amdgcn_grid_size_x() / __builtin_amdgcn_workgroup_size_x();
}

// Returns the number of workgroups in the 'y' dimension of the grid.
static inline uint32_t get_num_blocks_y() {
  return __builtin_amdgcn_grid_size_y() / __builtin_amdgcn_workgroup_size_y();
}

// Returns the number of workgroups in the 'z' dimension of the grid.
static inline uint32_t get_num_blocks_z() {
  return __builtin_amdgcn_grid_size_z() / __builtin_amdgcn_workgroup_size_z();
}

// Returns the total number of workgruops in the grid.
static inline uint64_t get_num_blocks() {
  return get_num_blocks_x() * get_num_blocks_y() * get_num_blocks_z();
}

// Returns the 'x' dimension of the current AMD workgroup's id.
static inline uint32_t get_block_id_x() {
  return __builtin_amdgcn_workgroup_id_x();
}

// Returns the 'y' dimension of the current AMD workgroup's id.
static inline uint32_t get_block_id_y() {
  return __builtin_amdgcn_workgroup_id_y();
}

// Returns the 'z' dimension of the current AMD workgroup's id.
static inline uint32_t get_block_id_z() {
  return __builtin_amdgcn_workgroup_id_z();
}

// Returns the absolute id of the AMD workgroup.
static inline uint64_t get_block_id() {
  return get_block_id_x() + get_num_blocks_x() * get_block_id_y() +
         get_num_blocks_x() * get_num_blocks_y() * get_block_id_z();
}

// Returns the number of workitems in the 'x' dimension.
static inline uint32_t get_num_threads_x() {
  return __builtin_amdgcn_workgroup_size_x();
}

// Returns the number of workitems in the 'y' dimension.
static inline uint32_t get_num_threads_y() {
  return __builtin_amdgcn_workgroup_size_y();
}

// Returns the number of workitems in the 'z' dimension.
static inline uint32_t get_num_threads_z() {
  return __builtin_amdgcn_workgroup_size_z();
}

// Returns the total number of workitems in the workgroup.
static inline uint64_t get_num_threads() {
  return get_num_threads_x() * get_num_threads_y() * get_num_threads_z();
}

// Returns the 'x' dimension id of the workitem in the current AMD workgroup.
static inline uint32_t get_thread_id_x() {
  return __builtin_amdgcn_workitem_id_x();
}

// Returns the 'y' dimension id of the workitem in the current AMD workgroup.
static inline uint32_t get_thread_id_y() {
  return __builtin_amdgcn_workitem_id_y();
}

// Returns the 'z' dimension id of the workitem in the current AMD workgroup.
static inline uint32_t get_thread_id_z() {
  return __builtin_amdgcn_workitem_id_z();
}

// Returns the absolute id of the thread in the current AMD workgroup.
static inline uint64_t get_thread_id() {
  return get_thread_id_x() + get_num_threads_x() * get_thread_id_y() +
         get_num_threads_x() * get_num_threads_y() * get_thread_id_z();
}

// Returns the size of an AMD wavefront, either 32 or 64 depending on hardware
// and compilation options.
static inline uint32_t get_lane_size() {
  return __builtin_amdgcn_wavefrontsize();
}

// Returns the id of the thread inside of an AMD wavefront executing together.
[[clang::convergent]] static inline uint32_t get_lane_id() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

// Returns the bit-mask of active threads in the current wavefront.
[[clang::convergent]] static inline uint64_t get_lane_mask() {
  return __builtin_amdgcn_read_exec();
}

// Copies the value from the first active thread in the wavefront to the rest.
[[clang::convergent]] static inline uint32_t broadcast_value(uint64_t mask,
                                                             uint32_t x) {
  return __builtin_amdgcn_readfirstlane(x);
}

// Returns a bitmask of threads in the current lane for which \p x is true.
[[clang::convergent]] static inline uint64_t ballot(uint64_t lane_mask,
                                                    _Bool x) {
  // the lane_mask & gives the nvptx semantics when lane_mask is a subset of
  // the active threads
  return lane_mask & __builtin_amdgcn_ballot_w64(x);
}

// Waits for all the threads in the block to converge and issues a fence.
[[clang::convergent]] static inline void sync_threads() {
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// Waits for all pending memory operations to complete in program order.
[[clang::convergent]] static inline void memory_fence() {
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "");
}

// Wait for all threads in the wavefront to converge, this is a noop on AMDGPU.
[[clang::convergent]] static inline void sync_lane(uint64_t mask) {
  __builtin_amdgcn_wave_barrier();
}

// Shuffles the the lanes inside the wavefront according to the given index.
[[clang::convergent]] static inline uint32_t shuffle(uint64_t mask,
                                                     uint32_t idx, uint32_t x) {
  return __builtin_amdgcn_ds_bpermute(idx << 2, x);
}

// Get the first active thread inside the lane.
static inline uint64_t get_first_lane_id(uint64_t lane_mask) {
  return __builtin_ffsll(lane_mask) - 1;
}

// Conditional that is only true for a single thread in a lane.
static inline _Bool is_first_lane(uint64_t lane_mask) {
  return get_lane_id() == get_first_lane_id(lane_mask);
}

#endif
