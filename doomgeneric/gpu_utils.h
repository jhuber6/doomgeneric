#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <stdint.h>

// Type aliases to the address spaces used by the AMDGPU backend.
#define Private [[clang::opencl_private]]
#define Constant [[clang::opencl_constant]]
#define Local [[clang::opencl_local]]
#define Global [[clang::opencl_global]]

#if defined(__AMDGPU__)
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

#elif defined(__NVPTX__)

// Returns the number of CUDA blocks in the 'x' dimension.
static inline uint32_t get_num_blocks_x() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}

// Returns the number of CUDA blocks in the 'y' dimension.
static inline uint32_t get_num_blocks_y() {
  return __nvvm_read_ptx_sreg_nctaid_y();
}

// Returns the number of CUDA blocks in the 'z' dimension.
static inline uint32_t get_num_blocks_z() {
  return __nvvm_read_ptx_sreg_nctaid_z();
}

// Returns the 'x' dimension of the current CUDA block's id.
static inline uint32_t get_block_id_x() {
  return __nvvm_read_ptx_sreg_ctaid_x();
}

// Returns the 'y' dimension of the current CUDA block's id.
static inline uint32_t get_block_id_y() {
  return __nvvm_read_ptx_sreg_ctaid_y();
}

// Returns the 'z' dimension of the current CUDA block's id.
static inline uint32_t get_block_id_z() {
  return __nvvm_read_ptx_sreg_ctaid_z();
}

// Returns the number of CUDA threads in the 'x' dimension.
static inline uint32_t get_num_threads_x() {
  return __nvvm_read_ptx_sreg_ntid_x();
}

// Returns the number of CUDA threads in the 'y' dimension.
static inline uint32_t get_num_threads_y() {
  return __nvvm_read_ptx_sreg_ntid_y();
}

// Returns the number of CUDA threads in the 'z' dimension.
static inline uint32_t get_num_threads_z() {
  return __nvvm_read_ptx_sreg_ntid_z();
}

// Returns the 'x' dimension id of the thread in the current CUDA block.
static inline uint32_t get_thread_id_x() {
  return __nvvm_read_ptx_sreg_tid_x();
}

// Returns the 'y' dimension id of the thread in the current CUDA block.
static inline uint32_t get_thread_id_y() {
  return __nvvm_read_ptx_sreg_tid_y();
}

// Returns the 'z' dimension id of the thread in the current CUDA block.
static inline uint32_t get_thread_id_z() {
  return __nvvm_read_ptx_sreg_tid_z();
}

// Returns the size of a CUDA warp, always 32 on NVIDIA hardware.
static inline uint32_t get_lane_size() { return 32; }

// Returns the id of the thread inside of a CUDA warp executing together.
[[clang::convergent]] static inline uint32_t get_lane_id() {
  return __nvvm_read_ptx_sreg_laneid();
}

// Returns the bit-mask of active threads in the current warp.
[[clang::convergent]] static inline uint64_t get_lane_mask() {
  return __nvvm_activemask();
}

// Copies the value from the first active thread in the warp to the rest.
[[clang::convergent]] static inline uint32_t broadcast_value(uint64_t lane_mask,
                                                             uint32_t x) {
  uint32_t mask = static_cast<uint32_t>(lane_mask);
  uint32_t id = __builtin_ffs(mask) - 1;
  return __nvvm_shfl_sync_idx_i32(mask, x, id, get_lane_size() - 1);
}

// Returns a bitmask of threads in the current lane for which \p x is true.
[[clang::convergent]] static inline uint64_t ballot(uint64_t lane_mask,
                                                    _Bool x) {
  uint32_t mask = static_cast<uint32_t>(lane_mask);
  return __nvvm_vote_ballot_sync(mask, x);
}

// Waits for all the threads in the block to converge and issues a fence.
[[clang::convergent]] static inline void sync_threads() { __syncthreads(); }

// Waits for all threads in the warp to reconverge for independent scheduling.
[[clang::convergent]] static inline void sync_lane(uint64_t mask) {
  __nvvm_bar_warp_sync(static_cast<uint32_t>(mask));
}

// Shuffles the the lanes inside the warp according to the given index.
[[clang::convergent]] static inline uint32_t shuffle(uint64_t lane_mask,
                                                     uint32_t idx, uint32_t x) {
  uint32_t mask = static_cast<uint32_t>(lane_mask);
  uint32_t bitmask = (mask >> idx) & 1;
  return -bitmask & __nvvm_shfl_sync_idx_i32(mask, x, idx, get_lane_size() - 1);
}

#else

static inline uint32_t get_num_blocks_x() { return 1; }

static inline uint32_t get_num_blocks_y() { return 1; }

static inline uint32_t get_num_blocks_z() { return 1; }

static inline uint32_t get_block_id_x() { return 0; }

static inline uint32_t get_block_id_y() { return 0; }

static inline uint32_t get_block_id_z() { return 0; }

static inline uint32_t get_num_threads_x() { return 1; }

static inline uint32_t get_num_threads_y() { return 1; }

static inline uint32_t get_num_threads_z() { return 1; }

static inline uint32_t get_thread_id_x() { return 0; }

static inline uint32_t get_thread_id_y() { return 0; }

static inline uint32_t get_thread_id_z() { return 0; }

static inline uint32_t get_lane_size() { return 1; }

static inline uint32_t get_lane_id() { return 0; }

static inline uint64_t get_lane_mask() { return 1; }

static inline uint32_t broadcast_value(uint64_t, uint32_t x) { return x; }

static inline uint64_t ballot(uint64_t, _Bool x) { return x; }

static inline void sync_threads() {}

static inline void sync_lane(uint64_t) {}

static inline uint32_t shuffle(uint64_t, uint32_t, uint32_t x) { return x; }

#endif

// Returns the absolute id of the thread in the current AMD workgroup.
static inline uint64_t get_thread_id() {
  return get_thread_id_x() + get_num_threads_x() * get_thread_id_y() +
         get_num_threads_x() * get_num_threads_y() * get_thread_id_z();
}

// Returns the total number of workgruops in the grid.
static inline uint64_t get_num_blocks() {
  return get_num_blocks_x() * get_num_blocks_y() * get_num_blocks_z();
}

// Returns the total number of workitems in the workgroup.
static inline uint64_t get_num_threads() {
  return get_num_threads_x() * get_num_threads_y() * get_num_threads_z();
}

/// Returns the absolute id of the CUDA block.
static inline uint64_t get_block_id() {
  return get_block_id_x() + get_num_blocks_x() * get_block_id_y() +
         get_num_blocks_x() * get_num_blocks_y() * get_block_id_z();
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
