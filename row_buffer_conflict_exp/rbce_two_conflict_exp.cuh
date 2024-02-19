#include "stdint.h"
#ifndef GPU_ROWHAMMER_RBCE_TWO_CONFLICT_EXP_H
#define GPU_ROWHAMMER_RBCE_TWO_CONFLICT_EXP_H

void init_rbce_two_conflict_exp();

__forceinline__ __device__ void
time_access(uint64_t *addr_access, uint64_t *addr_other, uint64_t *time_arr)
{
  uint64_t temp __attribute__((unused)), clock_start, clock_end;

  asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" ::"l"(addr_access));

  __syncthreads();
  clock_start = clock64();
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr_access));
  clock_end = clock64();

  *time_arr = clock_end - clock_start;
  // printf("%ld %p %p %ld\n", temp, addr_access, addr_other, *time_arr);
}

#endif /* GPU_ROWHAMMER_RBCE_TWO_CONFLICT_EXP_H */