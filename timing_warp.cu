#include "timing_util.h"
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <tuple>
#include <unistd.h>
const uint64_t EXP_COUNT = 10;
const uint64_t STEP_SIZE = 8; /* In Bytes */
const uint64_t LAYOUT_SIZE_1GB = 16106127360;
// const uint64_t TIME_ARR_SIZE = LAYOUT_SIZE_1GB / STEP_SIZE;

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
  // printf("%ld %p %p %ld\n", temp, addr_access, addr_other, *time_arr);
}

__forceinline__ __device__ void time_access_triple(uint64_t *addr_access,
                                                   uint64_t *addr_o1,
                                                   uint64_t *addr_o2,
                                                   uint64_t *time_arr)
{
  uint64_t temp, clock_start, clock_end;

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
  printf("%ld %p %ld\n", threadIdx.x, addr_access, *time_arr);
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
  printf("%ld\n", addr_access);
  time_access_triple(addr_access, addr_o1, addr_o2, time_arr + threadIdx.x);
}

std::tuple<uint64_t, uint64_t, uint64_t>
repeat_exp(uint64_t *addr, uint64_t *fixed_addr, uint64_t ofs,
           uint64_t *time_arr, uint64_t it, std::ofstream &file)
{
  uint64_t *time_arr_res = (uint64_t *)malloc(2 * sizeof(time_arr_res) * it);
  for (uint64_t i = 0; i < it; i++)
  {
    time_addr_kernel<<<1, 2>>>(addr, fixed_addr, ofs, time_arr + 2 * i);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(time_arr_res, time_arr, 2 * sizeof(time_arr_res) * it,
             cudaMemcpyDeviceToHost);

  uint64_t avg1 = 0, avg2 = 0, min = *(time_arr_res);
  for (uint64_t i = 0; i < it; i++)
  {
    avg1 += *(time_arr_res + 2 * i);
    avg2 += *(time_arr_res + 2 * i + 1);
    min = *(time_arr_res + 2 * i) < min ? *(time_arr_res + 2 * i) : min;
    file << fixed_addr << '\t' << addr + ofs << '\t' << ofs * 8 << '\t'
         << *(time_arr_res + 2 * i) << '\t' << *(time_arr_res + 2 * i + 1)
         << '\n';
  }

  avg1 /= it;
  avg2 /= it;

  // std::cout << avg1 << '\n';
  return std::make_tuple<>(avg1, avg2, min);
}

std::tuple<uint64_t *, uint64_t *, uint64_t *>
three_conflict_exp(uint64_t *addr, uint64_t *time_arr, uint64_t it)
{
  std::ofstream myfile;
  myfile.open("o16.txt");

  uint64_t step = 0;
  for (; step < 10485760 / 8; step += STEP_SIZE)
  {
    auto time_tup = repeat_exp(addr, addr, step, time_arr, EXP_COUNT, myfile);
    if (std::get<2>(time_tup) >= 1040)
      break;
  }

  uint64_t conflict_step1 = step;
  for (step = 0; step < 10485760 / 8; step += STEP_SIZE)
  {
    auto time_tup1 = repeat_exp(addr, addr + conflict_step1, step, time_arr,
                                EXP_COUNT, myfile);
    auto time_tup2 = repeat_exp(addr, addr, step, time_arr, EXP_COUNT, myfile);
    if (std::get<2>(time_tup1) >= 1040 && std::get<2>(time_tup2) >= 1040)
    {
      std::cout << std::get<2>(time_tup1) << "\t" << std::get<2>(time_tup2)
                << '\n';
      break;
    }
  }

  myfile.close();
  std::cout << conflict_step1 << "\t" << step << '\n';

  return std::make_tuple<>(addr, addr + conflict_step1, addr + step);
}

int main(void)
{
  uint64_t *addr;
  cudaMalloc(&addr, LAYOUT_SIZE_1GB);

  uint64_t *time_arr;
  cudaMalloc(&time_arr, 2 * sizeof(time_arr) * EXP_COUNT);
  time_addr_kernel<<<1, 2>>>(addr, addr, 0, time_arr);
  cudaDeviceSynchronize();

  // std::ofstream myfile;
  // myfile.open("o16.txt");

  // uint64_t step = 0;
  // for (; step < 10485760 / 8; step += STEP_SIZE)
  // {
  //   auto time_tup = repeat_exp(addr, addr, step, time_arr, EXP_COUNT,
  //   myfile); if (std::get<2>(time_tup) >= 1040)
  //     break;
  // }
  // myfile.close();

  // void *conflict = addr + step;
  // std::cout << "OK";
  // std::cin >> step;
  // myfile.open("o16.txt");
  // for (step = 0; step < 10485760 / 8; step += STEP_SIZE)
  // {
  //   auto time_tup = repeat_exp(addr, (uint64_t *)conflict, step, time_arr,
  //                              EXP_COUNT, myfile);
  //   // if (std::get<0>(time_tup) >= 1040)
  //   //   break;
  // }
  // myfile.close();
  auto conflict_tup = three_conflict_exp(addr, time_arr, 10);
  // three_conflict_kernel<<<1, 3>>>(std::get<0>(conflict_tup),
  //                                 std::get<1>(conflict_tup),
  //                                 std::get<2>(conflict_tup), time_arr);
  three_conflict_kernel<<<1, 3>>>(addr, addr + 1, addr + 2, time_arr);
  std::cout << toNS(1100) << '\t' << toNS(1040) << '\n';
  cudaDeviceSynchronize();
  struct cudaDeviceProp ok;
  cudaGetDeviceProperties(&ok, 0);
  std::cout << ok.clockRate << '\n';
}