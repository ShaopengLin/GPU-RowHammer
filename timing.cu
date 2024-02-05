#include <assert.h>
#include <iostream>
#include <memory>
#include <tuple>
#include <unistd.h>
#include <vector>
const uint64_t LAYOUT_SIZE = 16106127360; // 1073741824; //16106127360;

long double toNS(long long time)
{
  return (time / 1620000000.0f) * 1000000000.0;
}
__global__ void timing_pair_kernel(uint64_t *target, uint64_t *it_addr,
                                   long long *time)
{
  volatile long long clock_start;
  volatile long long clock_end;
  volatile uint64_t temp;

  /* Bring iterator to row buffer and uncache it. */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "discard.global.L2 [%1], 128;\n\t"
               "}"
               : "=l"(temp)
               : "l"(it_addr));

  /* Uncache target from previous accesses */
  asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}"
               : "+l"(target));

  /* Start Clock (Note: putting discard together with this will cause it
     to be optimized away) */
  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "}"
               : "=l"(clock_start));

  /* Test access to target. If it_addr is not a conflict should be fast */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(target));

  /* End clock */
  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "}"
               : "=l"(clock_end));

  *time = clock_end - clock_start;
}

__global__ void timing_triple_kernel(uint64_t *addr1, uint64_t *addr2,
                                     uint64_t *addr3, long long *time1,
                                     long long *time2)
{
  volatile long long clock_start;
  volatile long long clock_end;
  volatile uint64_t temp;
  volatile uint64_t temp1;
  /* Bring iterator to row buffer and uncache it. */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "discard.global.L2 [%1], 128;\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr1));

  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "discard.global.L2 [%1], 128;\n\t"
               "}"
               : "=l"(temp1)
               : "l"(addr3));

  /* Uncache target from previous accesses */
  asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}"
               : "+l"(addr2));

  /* Start Clock (Note: putting discard together with this will cause it
   to be optimized away) */
  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "}"
               : "=l"(clock_start));

  /* Test access to target. If it_addr is not a conflict should be fast */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr2));

  /* End clock */
  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "}"
               : "=l"(clock_end));

  *time1 = clock_end - clock_start;

  /* Start Clock (Note: putting discard together with this will cause it
     to be optimized away) */
  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "}"
               : "=l"(clock_start));

  /* Test access to target. If it_addr is not a conflict should be fast */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp1)
               : "l"(addr3));

  /* End clock */
  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "}"
               : "=l"(clock_end));

  *time2 = clock_end - clock_start;
}

long double test_single_pair(uint64_t *target, uint64_t *it_addr,
                             uint32_t test_it, long long *time)
{
  long double avg_time = 0;
  for (int i = 0; i < test_it; i++)
  {
    timing_pair_kernel<<<1, 1>>>(target, it_addr, time);
    cudaDeviceSynchronize();
    avg_time += *time;
  }
  avg_time /= test_it;

  return toNS(avg_time);
}

std::tuple<long double, long double>
test_single_triple(uint64_t *addr1, uint64_t *addr2, uint64_t *addr3,
                   uint32_t test_it, long long *time1, long long *time2)
{
  long double avg_time1 = 0;
  long double avg_time2 = 0;
  for (int i = 0; i < test_it; i++)
  {
    timing_triple_kernel<<<1, 1>>>(addr1, addr2, addr3, time1, time2);
    cudaDeviceSynchronize();
    avg_time1 += *time1;
    avg_time2 += *time2;
  }
  avg_time1 /= test_it;
  avg_time2 /= test_it;
  return std::make_tuple(toNS(avg_time1), toNS(avg_time2));
}

/**
 * @brief Find two conflict first, then go through rest that finds one that
 * conflict with both
 *
 * @param addr address space
 * @param threshold threshold to be considered a conflict
 */
std::vector<uint64_t *> find_n_conflict(uint64_t *addr, uint64_t n,
                                        long long threshold)
{

  /* Init host side time */
  long long *time;
  cudaHostAlloc(&time, sizeof(long long), cudaHostAllocDefault);
  std::vector<uint64_t *> conflict_lst{addr};

  /* First ever access has higher delay for some reason, avoid this outlier. */
  timing_pair_kernel<<<1, 1>>>(addr, addr, time);

  /* Find first outlier */
  long long i = 0;
  for (; i < LAYOUT_SIZE / 8; i++)
  {
    bool is_conflict = true;
    for (const auto &conf_addr : conflict_lst)
      is_conflict &=
          test_single_pair(addr + i, conf_addr, 10, time) > threshold;

    if (is_conflict)
    {
      std::cout << addr + i << '\n';
      conflict_lst.push_back(addr + i);
    }

    if (conflict_lst.size() == n)
      break;
  }
  cudaFree(time);
  return conflict_lst;
}

void test_three_conflict_diff(std::vector<uint64_t *> conflict_lst) {}

int main(void)
{
  uint64_t *addr;
  cudaMalloc(&addr, LAYOUT_SIZE);

  long long *time;
  cudaHostAlloc(&time, sizeof(long long), cudaHostAllocDefault);

  long long *time2;
  cudaHostAlloc(&time2, sizeof(long long), cudaHostAllocDefault);

  long double ld_time;
  timing_pair_kernel<<<1, 1>>>(addr, addr, time);

  for (int i = 0; i < 100; i++)
  {
    ld_time = test_single_pair(addr, addr + i, 10, time);
    std::cout << addr + i << '\t' << ld_time << '\n';
  }

  auto conflict_lst = find_n_conflict(addr, 32, 400);
  // std::cout << test_single_pair(conflict_lst[0], conflict_lst[1], 10, time)
  //           << '\n';
  // std::cout << test_single_pair(conflict_lst[1], conflict_lst[2], 10, time)
  //           << '\n';
  // timing_triple_kernel<<<1, 1>>>(conflict_lst[0], conflict_lst[1],
  //                                conflict_lst[2], time, time2);
  // cudaDeviceSynchronize();
  // auto tup = test_single_triple(conflict_lst[0], conflict_lst[1],
  //                               conflict_lst[2], 10, time, time2);
  // std::cout << std::get<0>(tup) << '\n';
  // std::cout << std::get<1>(tup) << '\n';

  cudaFree(addr);
  cudaFree(time);
}
