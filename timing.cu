#include <assert.h>
#include <iostream>
#include <unistd.h>
const uint64_t LAYOUT_SIZE = 16106127360; // 1073741824; //16106127360;

__global__ void mapping_kernel(uint64_t *target, uint64_t *it_addr,
                               long long *time) {
  volatile long long clock_start;
  volatile long long clock_end;
  volatile uint64_t temp;

  /* Bring iterator to row buffer and uncache it. */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "fence.sc.cta;\n\t"
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
               "fence.sc.cta;\n\t"
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
               "fence.sc.cta;\n\t"
               "}"
               : "=l"(clock_end));

  *time = clock_end - clock_start;
}

long double test_single_pair(uint64_t *target, uint64_t *it_addr,
                             uint32_t test_it, long long *time) {
  long double avg_time = 0;
  for (int i = 0; i < test_it; i++) {
    mapping_kernel<<<1, 1>>>(target, it_addr, time);
    cudaDeviceSynchronize();
    avg_time += *time;
  }
  avg_time /= 10;
  avg_time = (avg_time / 1620000000.0f) * 1000000000.0;

  return avg_time;
}

int main(void) {
  uint64_t *addr;
  cudaMalloc(&addr, LAYOUT_SIZE);

  long long *time;
  cudaHostAlloc(&time, sizeof(long long), cudaHostAllocDefault);

  uint64_t *conflict_addr = NULL;
  long double ld_time;
  for (int i = 0; i < 1024 * 10; i++) {

    ld_time = test_single_pair(addr, addr + i, 10, time);
    std::cout << addr + i << '\t' << ld_time << '\n';

    if (ld_time > 400 && i > 0) {
      conflict_addr = addr + i;
      break;
    }
  }

  std::cout << "\nFound Row Buffer Conflict:" << '\n';
  std::cout << "\nAccess " << addr << " then " << conflict_addr << '\n'
            << test_single_pair(conflict_addr, addr, 10, time) << '\n';
  std::cout << "\nAccess " << conflict_addr << " then " << addr << '\n'
            << test_single_pair(addr, conflict_addr, 10, time) << '\n';

  std::cout << "\nRow Buffered Accesses to " << addr << '\n'
            << test_single_pair(addr, addr, 10, time) << '\n';

  std::cout << "\nRow Buffered Accesses to " << conflict_addr  << '\n'
            << test_single_pair(conflict_addr, conflict_addr, 10, time) << '\n';

  std::cout << "\nAccess " << conflict_addr << " then " << conflict_addr + 1
            << '\n'
            << test_single_pair(conflict_addr, conflict_addr + 1, 10, time)
            << '\n';
  std::cout << "\nAccess " << conflict_addr + 1 << " then " << conflict_addr
            << '\n'
            << test_single_pair(conflict_addr + 1, conflict_addr, 10, time)
            << '\n';

  cudaFree(addr);
}
