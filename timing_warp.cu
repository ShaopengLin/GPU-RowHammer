#include "timing_util.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdint.h>
#include <stdio.h>
#include <tuple>
#include <unistd.h>
#include <vector>
constexpr uint64_t EXP_COUNT = 10;
const uint64_t STEP_SIZE = 8; /* In Bytes */
const uint64_t LAYOUT_SIZE_1GB = 16106127360;
const uint64_t CONFLICT_TRSHOLD = 570;
static uint64_t time_arr_host[2 * EXP_COUNT];
// const uint64_t TIME_ARR_SIZE = LAYOUT_SIZE_1GB / STEP_SIZE;

__forceinline__ __device__ void
time_access(uint64_t *addr_access, uint64_t *addr_other, uint64_t *time_arr)
{
  uint64_t temp __attribute__((unused)), clock_start, clock_end;

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
  // printf("%ld %p %p %ld\n", temp, addr_access, addr_other, *time_arr);
}

__forceinline__ __device__ void time_access_triple(uint64_t *addr_access,
                                                   uint64_t *addr_o1,
                                                   uint64_t *addr_o2,
                                                   uint64_t *time_arr)
{
  uint64_t temp __attribute__((unused)), clock_start, clock_end;

  asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "discard.global.L2 [%1], 128;\n\t"
               "discard.global.L2 [%2], 128;\n\t"
               "}" ::"l"(addr_access),
               "l"(addr_o1), "l"(addr_o2));

  __syncthreads();
  clock_start = clock64();
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr_access));
  clock_end = clock64();

  *time_arr = clock_end - clock_start;
  printf("%d %p %ld\n", threadIdx.x, addr_access, *time_arr);
}

__global__ void time_addr_kernel(uint64_t *addr, uint64_t *fixed_addr,
                                 uint64_t ofs, uint64_t *time_arr)
{
  uint64_t *addr_access = threadIdx.x == 0 ? fixed_addr : addr + ofs;
  uint64_t *addr_other = threadIdx.x == 0 ? addr + ofs : fixed_addr;

  time_access(addr_access, addr_other, time_arr + threadIdx.x);
}

__global__ void three_conflict_kernel(uint64_t *addr1, uint64_t *addr2,
                                      uint64_t *addr3, uint64_t *time_arr)
{
  uint64_t *addr_access;
  uint64_t *addr_o1;
  uint64_t *addr_o2;
  switch (threadIdx.x)
  {
  case 0:
    addr_access = addr1;
    addr_o1 = addr2;
    addr_o2 = addr3;
    break;
  case 1:
    addr_access = addr2;
    addr_o1 = addr1;
    addr_o2 = addr3;
    break;
  case 2:
    addr_access = addr3;
    addr_o1 = addr1;
    addr_o2 = addr2;
    break;
  default:
    break;
  }
  time_access_triple(addr_access, addr_o1, addr_o2, time_arr + threadIdx.x);
}

std::tuple<uint64_t, uint64_t> repeat_exp(uint64_t *addr, uint64_t *fixed_addr,
                                          uint64_t ofs, uint64_t *time_arr,
                                          uint64_t it, std::ofstream *file,
                                          bool do_print = false)
{
  for (uint64_t i = 0; i < it; i++)
    time_addr_kernel<<<1, 2>>>(addr, fixed_addr, ofs, time_arr + 2 * i);
  cudaDeviceSynchronize();
  cudaMemcpy(time_arr_host, time_arr, 2 * sizeof(uint64_t *) * it,
             cudaMemcpyDeviceToHost);

  uint64_t avg = 0, min = LONG_MAX;
  for (uint64_t i = 0; i < it; i++)
  {
    avg += time_arr_host[2 * i];
    min = std::min(time_arr_host[2 * i], min);
  }

  avg /= it;

  if (do_print)
    *file << fixed_addr << '\t' << min << '\n';

  return std::make_tuple<>(avg, min);
}

std::vector<uint64_t *> n_conflict_exp(uint64_t *addr, uint64_t *time_arr,
                                       uint64_t n, uint64_t it)
{
  std::vector<uint64_t *> conf_vec{addr};
  uint64_t step = 0;
  while (conf_vec.size() < n)
  {
    for (; step < LAYOUT_SIZE_1GB / 8; step += STEP_SIZE)
    {
      if (std::accumulate(
              conf_vec.begin(), conf_vec.end(), true,
              [addr, step, time_arr](bool all_conflict, uint64_t *conf_addr)
              {
                auto time_tup = repeat_exp(addr, conf_addr, step, time_arr,
                                           EXP_COUNT, NULL);
                return all_conflict &&
                       (std::get<1>(time_tup) > CONFLICT_TRSHOLD);
              }))
      {
        conf_vec.push_back(addr + step);
        std::cout << conf_vec.size() << " conflict: " << addr + step << '\n';
        break;
      }
    }
  }

  return conf_vec;
}
int main(void)
{
  uint64_t *addr;
  cudaMalloc(&addr, LAYOUT_SIZE_1GB);

  uint64_t *time_arr;
  cudaMalloc(&time_arr, 2 * sizeof(uint64_t) * EXP_COUNT);
  time_addr_kernel<<<1, 2>>>(addr, addr, 0, time_arr);

  std::ofstream myfile;
  myfile.open("o16.txt");

  uint64_t step = 0;
  for (; step < 419430400 / 8; step += STEP_SIZE)
    repeat_exp(addr, addr, step, time_arr, EXP_COUNT, &myfile, true);
  myfile.close();

  auto conf_vec = n_conflict_exp(addr, time_arr, 3, 10);
  three_conflict_kernel<<<1, 3>>>(conf_vec[0], conf_vec[1], conf_vec[2],
                                  time_arr);
  // std::cout << toNS(1100) << '\t' << toNS(1040) << '\n';
  // cudaDeviceSynchronize();
  // struct cudaDeviceProp ok;
  // cudaGetDeviceProperties(&ok, 0);
  // std::cout << ok.clockRate << '\n';
}