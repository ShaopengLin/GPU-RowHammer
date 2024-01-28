#include <iostream>
#include <assert.h>

const uint64_t LAYOUT_SIZE = 1048576; // 1073741824; //16106127360;

__global__ void mapping_kernel(uint16_t *target, uint16_t *it_addr, long long *time)
{
  uint16_t buf_t;
  uint16_t buf_it;

  // *it_addr = (uint16_t) it_addr;
  // /* Discard our target's cache from potential previous executions. */
  // asm("discard.global.L2 [%0], 128;" : "+l"(target));

  /* Bring it_addr to its row buffer in advance */
  // asm("ld.u16.global.volatile %0, [%1];" : "=h"(buf_it) : "l"(it_addr));

  // /* If it_addr is in the same bank as target, target will take longer due to conflict*/
  // asm volatile ("ld.u16.global.volatile %0, [%1];" : "=h"(buf_it) : "l"(target));
  asm volatile ("ld.u16.global.cv %0, [%1];" : "=h"(buf_it) : "l"(it_addr));
  asm("discard.global.L2 [%0], 128;" : "+l"(it_addr));
  asm("discard.global.L2 [%0], 128;" : "+l"(target));
  
  asm volatile ("ld.u16.global.volatile %0, [%1];" : "=h"(buf_t) : "l"(target));
  long long start = clock64();
  // asm volatile ("ld.u16.global.volatile %0, [%1];" : "=h"(buf_t) : "l"(target));
  asm volatile ("ld.u16.global.volatile %0, [%1];" : "=h"(buf_it) : "l"(it_addr));
  
  long long end = clock64();

  assert(buf_t == 0);
  assert(buf_it == 0);
  *time = end - start;
}
int main(void)
{
  uint16_t *d_x;
  cudaMalloc(&d_x, LAYOUT_SIZE);

  long long *time;
  cudaHostAlloc(&time, sizeof(long long), cudaHostAllocDefault);
  
  for (int i = 1; i < 20; i++)
  {
    mapping_kernel<<<1, 1>>>(d_x, d_x + i, time);
    cudaDeviceSynchronize();
    std::cout << *time << '\n';
  }
  // mapping_kernel<<<1, 1>>>(d_x, d_x, time);
  // cudaDeviceSynchronize();
  // std::cout << *time << '\n';
  size_t f, t;
  cudaMemGetInfo(&f, &t);
  std::cout << f << '\n';
  std::cout << t << '\n';
  std::cout << d_x << '\n';
  cudaFree(d_x);
}