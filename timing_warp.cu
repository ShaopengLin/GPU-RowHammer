#include "timing_util.h"
#include <stdint.h>
#include <stdio.h>
#include <tuple>
#include <unistd.h>
const uint64_t STEP_SIZE = 32;
const uint64_t LAYOUT_SIZE_1GB = 1073741824;
const uint64_t TIME_ARR_SIZE = LAYOUT_SIZE_1GB / STEP_SIZE;

__forceinline__ __device__ void
time_access(uint64_t *addr_access, uint64_t *addr_other, uint64_t *time_arr)
{
  uint64_t temp, clock_start, clock_end;

  asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "discard.global.L2 [%1], 128;\n\t"
               "}" ::"l"(addr_access),
               "l"(addr_other));

  __syncthreads();
  clock_start = clock64();
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr_access));
  clock_end = clock64();

  *time_arr = clock_end - clock_start;
}

__global__ void time_addr_kernel(uint64_t *addr, uint64_t *fixed_addr,
                                 uint64_t ofs, uint64_t *time_arr)
{
  uint64_t *addr_access = threadIdx.x == 0 ? fixed_addr : addr + ofs;
  uint64_t *addr_other = threadIdx.x == 0 ? addr + ofs : fixed_addr;
  time_access(addr_access, addr_other, time_arr + threadIdx.x);
}

std::tuple<uint64_t, uint64_t> repeat_exp(uint64_t *addr, uint64_t *fixed_addr,
                                          uint64_t ofs, uint64_t *time_arr,
                                          uint64_t it)
{
  uint64_t *time_arr_res = (uint64_t *)malloc(2 * sizeof(time_arr_res) * it);
  for (uint64_t i = 0; i < it; i++)
    time_addr_kernel<<<1, 2>>>(addr, fixed_addr, ofs, time_arr + 2 * i);

  cudaMemcpy(time_arr_res, time_arr, 2 * sizeof(time_arr_res) * it,
             cudaMemcpyDeviceToHost);

  uint64_t avg1 = 0, avg2 = 0;
  for (uint64_t i = 0; i < it; i++)
  {
    avg1 += *(time_arr_res + 2 * i);
    avg2 += *(time_arr_res + 2 * i + 1);
  }

  avg1 /= it;
  avg2 /= it;
  printf("%Ld\t%Ld\n", toNS(avg1), toNS(avg2));

  return std::make_tuple<>(toNS(avg1), toNS(avg2));
}

int main(void)
{
  uint64_t *addr;
  cudaMalloc(&addr, LAYOUT_SIZE_1GB);

  uint64_t *time_arr;
  cudaMalloc(&time_arr, sizeof(time_arr) * TIME_ARR_SIZE);
  time_addr_kernel<<<1, 2>>>(addr, addr, 0, time_arr);

  for (uint64_t step = 0; step < LAYOUT_SIZE_1GB; step += STEP_SIZE / 4)
  {
    auto time_tup = repeat_exp(addr, addr, step, time_arr, 5);
    if (std::get<0>(time_tup) != std::get<1>(time_tup))
      break;
  }

  cudaDeviceSynchronize();
}