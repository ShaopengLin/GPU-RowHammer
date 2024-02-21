#include "timing_util.h"
#include <algorithm>
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
const uint64_t LAYOUT_SIZE_15GB = 16106127360;
const uint64_t CONFLICT_TRSHOLD = 1030;
const uint64_t SIZE_8MB = 8388608;
const uint64_t SIZE_400MB = 419430400;
static uint64_t time_arr_host[3 * EXP_COUNT];
// const uint64_t TIME_ARR_SIZE = LAYOUT_SIZE_1GB / STEP_SIZE;

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

__forceinline__ __device__ void time_access_triple(uint64_t *addr_access,
                                                   uint64_t *addr_o1,
                                                   uint64_t *addr_o2,
                                                   uint64_t *time_arr)
{
  uint64_t temp __attribute__((unused)), clock_start, clock_end;

  asm volatile("{\n\t"
               //  "discard.global.L2 [%0], 128;\n\t"
               //  "discard.global.L2 [%0], 128;\n\t"
               "discard.global.L2 [%0], 128;\n\t"
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

__global__ void time_addr_kernel(uint64_t *addr1, uint64_t *addr2,
                                 uint64_t *time_arr)
{
  uint64_t *addr_access = threadIdx.x == 0 ? addr2 : addr1;
  uint64_t *addr_other = threadIdx.x == 0 ? addr1 : addr2;

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

std::tuple<uint64_t, uint64_t> repeat_exp(uint64_t *addr1, uint64_t *addr2,
                                          uint64_t *time_arr, uint64_t it,
                                          std::ofstream *file,
                                          bool do_print = false)
{
  for (uint64_t i = 0; i < it; i++)
    time_addr_kernel<<<1, 2>>>(addr1, addr2, time_arr + 2 * i);
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
    *file << '(' << addr1 << ", " << addr2 << ")\t" << toNS(min) << '\n';

  return std::make_tuple<>(avg, min);
}

std::tuple<uint64_t, uint64_t>
repeat_exp_3conflict(uint64_t *addr1, uint64_t *addr2, uint64_t *addr3,
                     uint64_t *time_arr, uint64_t it, std::ofstream *file,
                     bool do_print = false)
{
  for (uint64_t i = 0; i < it; i++)
    three_conflict_kernel<<<1, 3>>>(addr1, addr2, addr3, time_arr + 3 * i);
  cudaDeviceSynchronize();
  cudaMemcpy(time_arr_host, time_arr, 3 * sizeof(uint64_t *) * it,
             cudaMemcpyDeviceToHost);
  uint64_t avg = 0, min = LONG_MAX;
  for (uint64_t i = 0; i < it; i++)
  {
    avg += time_arr_host[3 * i];
    min = std::min(time_arr_host[3 * i], min);
  }

  avg /= it;

  if (do_print)
    *file << '(' << addr1 << ", " << addr2 << ", " << addr3 << ")\t"
          << toNS(min) << "ns" << '\n';

  return std::make_tuple<>(avg, min);
}

std::vector<uint64_t *> n_conflict_exp(uint64_t *addr, uint64_t *time_arr,
                                       uint64_t n, uint64_t it)
{
  std::vector<uint64_t *> conf_vec{addr};
  uint64_t step = 0;
  while (conf_vec.size() < n)
  {
    for (; step < LAYOUT_SIZE_15GB / 8; step += STEP_SIZE)
    {
      if (std::accumulate(
              conf_vec.begin(), conf_vec.end(), true,
              [addr, step, time_arr](bool all_conflict, uint64_t *conf_addr)
              {
                auto time_tup = repeat_exp(addr + step, conf_addr, time_arr,
                                           EXP_COUNT, NULL);
                return all_conflict &&
                       (std::get<1>(time_tup) > CONFLICT_TRSHOLD);
              }))
      {
        conf_vec.push_back(addr + step);
        break;
      }
    }
  }

  uint64_t *initial = conf_vec[0];
  for (auto it = conf_vec.begin() + 1; it != conf_vec.end(); it++)
  {
    std::cout << " Conflict: " << *it << ", Offset: " << (*it - initial) / 256
              << '\n';
    initial = *it;
  }

  return conf_vec;
}

void compare_2and3_conflict_exp(std::vector<uint64_t *> &conf_set,
                                uint64_t *time_arr, uint64_t it,
                                std::ofstream *file)
{
  {
    std::string bitmask(3, 1);          // K leading 1's
    bitmask.resize(conf_set.size(), 0); // N-K trailing 0's
    uint64_t *addr_arr[3];              // Storage of conflicts
    // print integers and permute bitmask
    do
    {
      int addr_added = 0;
      for (int i = 0; i < conf_set.size(); ++i) // [0..N-1] integers
      {
        if (bitmask[i])
        {
          addr_arr[addr_added] = conf_set[i];
          addr_added++;
        }
      }
      repeat_exp_3conflict(addr_arr[0], addr_arr[1], addr_arr[2], time_arr,
                           EXP_COUNT, file, true);
      std::cout << '\n';
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
  }

  {
    std::string bitmask(2, 1);          // K leading 1's
    bitmask.resize(conf_set.size(), 0); // N-K trailing 0's
    uint64_t *addr_arr[2];              // Storage of conflicts
    // print integers and permute bitmask
    do
    {
      int addr_added = 0;
      for (int i = 0; i < conf_set.size(); ++i) // [0..N-1] integers
      {
        if (bitmask[i])
        {
          addr_arr[addr_added] = conf_set[i];
          addr_added++;
        }
      }
      repeat_exp(addr_arr[0], addr_arr[1], time_arr, EXP_COUNT, file, true);
      std::cout << '\n';
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
  }
}

int main(void)
{
  uint64_t *addr;
  cudaMalloc(&addr, LAYOUT_SIZE_15GB);

  uint64_t *time_arr;
  cudaMalloc(&time_arr, 3 * sizeof(uint64_t) * EXP_COUNT);
  time_addr_kernel<<<1, 2>>>(addr, addr, time_arr);

  std::ofstream myfile;
  // myfile.open("8MB.txt");

  // uint64_t step = 0;
  // for (; step < SIZE_8MB / 8; step += STEP_SIZE)
  //   repeat_exp(addr, addr + step, time_arr, EXP_COUNT, &myfile, true);
  // myfile.close();

  // myfile.open("400MB.txt");

  // step = 0;
  // for (; step < SIZE_400MB / 8; step += STEP_SIZE)
  //   repeat_exp(addr, addr, step, time_arr, EXP_COUNT, &myfile, true);
  // myfile.close();

  myfile.open("3Conflict.txt");
  auto conf_vec = n_conflict_exp(addr, time_arr, 5, 10);
  compare_2and3_conflict_exp(conf_vec, time_arr, EXP_COUNT, &myfile);
  // repeat_exp_3conflict(conf_vec[0], conf_vec[1], conf_vec[2], time_arr,
  //                      EXP_COUNT, &myfile, true);
  // repeat_exp_3conflict(conf_vec[3], conf_vec[1], conf_vec[2], time_arr,
  //                      EXP_COUNT, &myfile, true);
  myfile.close();
  // three_conflict_kernel<<<1, 3>>>(conf_vec[0], conf_vec[1], conf_vec[2],
  //                                 time_arr);
  // cudaDeviceSynchronize();
  // three_conflict_kernel<<<1, 3>>>(conf_vec[0], conf_vec[1], conf_vec[2],
  //                                 time_arr);
  // cudaDeviceSynchronize();
  // three_conflict_kernel<<<1, 3>>>(conf_vec[0], conf_vec[1], conf_vec[2],
  //                                 time_arr);
  // std::cout << toNS(1095) << '\t' << toNS(1040) << toNS(980) << '\n';
  // cudaDeviceSynchronize();
  struct cudaDeviceProp ok;
  cudaGetDeviceProperties(&ok, 0);
  std::cout << ok.clockRate << '\n';
}