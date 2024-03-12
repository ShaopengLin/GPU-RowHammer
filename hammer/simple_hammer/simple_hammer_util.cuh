#include "stdint.h"
#include "stdio.h"
#ifndef GPU_ROWHAMMER_HAMMER_SIMPLE_HAMMER_SIMPLE_HAMMER_UTIL_CUH
#define GPU_ROWHAMMER_HAMMER_SIMPLE_HAMMER_SIMPLE_HAMMER_UTIL_CUH

/**
 * @brief Accesses the addr_access uncached in order to hit the row buffer.
 *
 * @param addr_access address access
 */
__forceinline__ __device__ void
uncached_access_device(uint64_t *addr_access)
{
  uint64_t temp __attribute__((unused));
  asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" ::"l"(addr_access));

  __syncthreads(); /* re-sync if divergence happend before this call */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr_access));
}

__forceinline__ __global__ void
n_address_conflict_kernel(uint64_t **addr_arr)
{
  uncached_access_device(*(addr_arr + threadIdx.x));
}

#endif /* GPU_ROWHAMMER_HAMMER_SIMPLE_HAMMER_SIMPLE_HAMMER_UTIL_CUH */