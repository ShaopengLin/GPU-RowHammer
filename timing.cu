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
  std::cout << it_addr << '\t' << avg_time << '\n';
  return avg_time;
}

int main(void) {
  uint64_t *d_x;
  cudaMalloc(&d_x, LAYOUT_SIZE);

  long long *time;
  cudaHostAlloc(&time, sizeof(long long), cudaHostAllocDefault);

  uint64_t *new_target = NULL;
  for (int i = 0; i < 1024 * 10; i++) {
    if (test_single_pair(d_x, d_x + i, 10, time) > 400 && i > 0) {
      new_target = d_x + i;
      break;
    }
  }

  std::cout << "\nFound Conflict:" << '\n';
  std::cout << test_single_pair(new_target, d_x, 10, time) << '\t'
            << test_single_pair(d_x, new_target, 10, time) << '\n';
  std::cout << "\nNew Target Normal Time:" << '\n';
  std::cout << test_single_pair(new_target, new_target, 10, time) << '\n';

  std::cout << "\nd_x Normal Time:" << '\n';
  std::cout << test_single_pair(new_target, new_target, 10, time) << '\n';

  cudaFree(d_x);
}
