#include <iostream>
#include <assert.h>
#include <unistd.h>
const uint64_t LAYOUT_SIZE = 16106127360; // 1073741824; //16106127360;

__global__ void mapping_kernel(uint64_t *target, uint64_t *it_addr, long long *time)
{
  volatile long long clock_start;
  volatile long long clock_end;
  volatile uint64_t temp;

  /* Bring iterator to row buffer and uncache it. */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "fence.sc.cta;\n\t"
               "discard.global.L2 [%1], 128;\n\t"
               "}" : "=l"(temp) : "l"(it_addr));

  /* Uncache target from previous accesses */
  asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" : "+l"(target));

  /* Start Clock (Note: putting discard together with this will cause it
     to be optimized away) */
  asm volatile("{\n\t"
               "fence.sc.cta;\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "}" : "=l"(clock_start));

  /* Test access to target. If it_addr is not a conflict should be fast */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}" : "=l"(temp) : "l"(target));

  /* End clock */
  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "fence.sc.cta;\n\t"
               "}" : "=l"(clock_end));

  *time = clock_end - clock_start;
}

int main(void)
{
  uint64_t *d_x;
  cudaMalloc(&d_x, LAYOUT_SIZE);

  long long *time;
  cudaHostAlloc(&time, sizeof(long long), cudaHostAllocDefault);

  for (int i = 0; i < 1024 * 1024; i++)
  {
    long long t = 0;
    for (int j = 0; j < 10; j++)
    {
      mapping_kernel<<<1, 1>>>(d_x, d_x + i, time);
      cudaDeviceSynchronize();
      t += *time;
      usleep(1);
    }
    t /= 10;
    std::cout << d_x + i << '\t' <<  (t/1620000000.0f)*1000000000.0 << '\n';
  }

  // size_t f, t;
  // cudaMemGetInfo(&f, &t);
  // std::cout << t - f << '\n';
  // std::cout << d_x << '\n';
  cudaFree(d_x);
}
