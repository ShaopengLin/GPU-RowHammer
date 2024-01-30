#include <iostream>
#include <assert.h>

const uint64_t LAYOUT_SIZE = 16106127360; // 1073741824; //16106127360 // 1048576 // 4294967296;

__global__ void mapping_kernel(uint64_t *target, uint64_t *it_addr, long long *time)
{
  uint64_t buf_t;
  uint64_t buf_it;

  asm volatile ("discard.global.L2 [%0], 128;" : "+l"(target));
  
  long long start = clock64();
  asm volatile ("ld.u64.global.volatile %0, [%1];" : "=l"(buf_t) : "l"(target));
  long long end = clock64();

  *time = end - start;
}
int main(void)
{
  uint64_t *d_x;
  cudaMalloc(&d_x, LAYOUT_SIZE);

  long long *time;
  cudaHostAlloc(&time, sizeof(long long), cudaHostAllocDefault);
  long long max = 0;
  for (int i = 1; i < 200; i++)
  {
      mapping_kernel<<<1, 1>>>(d_x, d_x + i, time);
      cudaDeviceSynchronize();
      std::cout << *time << '\n';
  }
  //std::cout << max << '\n';
  // mapping_kernel<<<1, 1>>>(d_x, d_x, time);
  // cudaDeviceSynchronize();
  // std::cout << *time << '\n';
  size_t f, t;
  cudaMemGetInfo(&f, &t);
  //std::cout << f << '\n';
  //std::cout << t << '\n';
  //std::cout << d_x << '\n';
  cudaFree(d_x);
}