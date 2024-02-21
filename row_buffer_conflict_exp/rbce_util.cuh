#include "stdint.h"
#include "stdio.h"
#ifndef GPU_ROWHAMMER_RBCE_UTIL_CUH
#define GPU_ROWHAMMER_RBCE_UTIL_CUH

namespace rbce
{

/**
 * @brief Stores the time of uncached access of addr_access in time_arr.
 * This function requires synchronized access with __syncthreads, please
 * make sure no divergence happends on places where this function is called.
 *
 * @param addr_access address top access
 * @param time_arr place to store timing valueW
 */
__forceinline__ __device__ void
uncached_access_timing_device(uint64_t *addr_access, uint64_t *time_arr)
{
  uint64_t temp __attribute__((unused)), clock_start, clock_end;
  asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" ::"l"(addr_access));

  __syncthreads(); /* re-sync if divergence happend before this call */
  clock_start = clock64();
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr_access));
  clock_end = clock64();
  *time_arr = clock_end - clock_start;
}
inline uint64_t toNS(uint64_t time, uint64_t clock_rate)
{
  return (time / 1545000000.0f) * 1000000000.0;
}

} // namespace rbce

#endif /* GPU_ROWHAMMER_RBCE_UTIL_CUH */