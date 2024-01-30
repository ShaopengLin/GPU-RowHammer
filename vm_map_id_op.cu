#include <iostream>
#include <assert.h>
#include <unistd.h>
const uint64_t LAYOUT_SIZE = 16106127360; // 1073741824; //16106127360;

__global__ void mapping_kernel(uint64_t *target, uint64_t *it_addr, long long *time)
{
  volatile long long clock_start;
  volatile long long clock_end;
  volatile uint64_t temp;
  volatile uint64_t temp1;
  /* Bring iterator to row buffer and uncache it. */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}" : "=l"(temp) : "l"(it_addr));

  asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" : "+l"(it_addr));

asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" : "+l"(target));
  /* Uncache our target (If our taget is in same row as it_addr, the result
     is the same. */
  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
              //  "discard.global.L2 [%1], 128;\n\t"
               "}" : "=l"(clock_start));

  /* Test access to target. If it_addr is not a conflict */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}" : "=l"(temp) : "l"(target));

  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "}" : "=l"(clock_end));
  // Clock

  // asm volatile("{\n\t"
  //              "discard.global.L2 [%0], 128;\n\t"
  //              "}" : "+l"(target));
  // asm volatile("{\n\t"
  //              "discard.global.L2 [%0], 128;\n\t"
  //              "}" : "+l"(it_addr));
  *time = clock_end - clock_start;

  // *time += clock_end - clock_start;
}

__global__ void target_time(uint64_t *target, long long *time)
{
  volatile long long clock_start;
  volatile long long clock_end;
  volatile uint64_t temp;
  /* Bring iterator to row buffer and uncache it. */

  /* Uncache our target (If our taget is in same row as it_addr, the result
     is the same. */
  asm volatile("{\n\t"
               "discard.global.L2 [%1], 128;\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "}" : "=l"(clock_start) : "l"(target));

  /* Test access to target. If it_addr is not a conflict */
  asm volatile("{\n\t"
               "ld.u64.global.volatile %0, [%1];\n\t"
               "}" : "=l"(temp) : "l"(target));

  asm volatile("{\n\t"
               "mov.u64 %0, %%clock64;\n\t"
               "}" : "=l"(clock_end));
  // Clock

  *time = clock_end - clock_start;

  // *time += clock_end - clock_start;
}

int main(void)
{
  uint64_t *d_x;
  cudaMalloc(&d_x, LAYOUT_SIZE);

  long long *time;
  cudaHostAlloc(&time, sizeof(long long), cudaHostAllocDefault);
  // mapping_kernel<<<1, 1>>>(d_x, d_x, time);
  // cudaDeviceSynchronize();
  // for (int i = 0; i < 1024 * 1024; i++)
  // {
  //   long long t = 0;
  //   for (int j = 0; j < 10; j++)
  //   {
  //     target_time<<<1, 1>>>(d_x, time);
  //     cudaDeviceSynchronize();
  //     t += *time;
  //     usleep(1);
  //   }
  //   t /= 10;
  //   std::cout << t << '\n';
  // }
  // std::cout << "True" << '\n';
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
    std::cout << t << '\n';
  }
  size_t f, t;
  cudaMemGetInfo(&f, &t);
  std::cout << t - f << '\n';
  std::cout << d_x << '\n';
  cudaFree(d_x);
}
