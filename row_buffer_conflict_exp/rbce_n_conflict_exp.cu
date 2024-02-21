#include "rbce_n_conflict_exp.cuh"

namespace rbce
{

__global__ void n_address_conflict_kernel(uint64_t **addr_arr,
                                          uint64_t *time_arr)
{
  uncached_access_timing_device(*(addr_arr + threadIdx.x),
                                time_arr + threadIdx.x);
}

N_Conflict::N_Conflict(int argc, char *argv[])
{
  if (argc >= 3)
    this->N = std::stoull(argv[2]);
  if (argc >= 4)
    this->EXP_RANGE = std::stoull(argv[3]);
  if (argc >= 5)
    this->EXP_IT = std::stoull(argv[4]);
  printf("%ld, %ld, %ld\n", this->N, this->EXP_RANGE, this->EXP_IT);
  cudaMalloc(&(this->ADDR_LAYOUT), this->LAYOUT_SIZE);
  cudaMalloc(&(this->TIME_ARR_DEVICE),
             this->N * sizeof(uint64_t) * this->EXP_IT);
  cudaMalloc(&(this->ADDR_LST_BUF), this->N * sizeof(uint64_t));
  this->TIME_ARR_HOST = new uint64_t[this->N * this->EXP_IT];

  struct cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  this->CLOCK_RATE = device_prop.clockRate;
}

N_Conflict::N_Conflict(uint64_t N, uint64_t EXP_RANGE, uint64_t EXP_IT)
    : N{N}, EXP_RANGE{EXP_RANGE}, EXP_IT{EXP_IT}
{
  printf("%ld, %ld, %ld\n", this->N, this->EXP_RANGE, this->EXP_IT);
  cudaMalloc(&(this->ADDR_LAYOUT), this->LAYOUT_SIZE);
  cudaMalloc(&(this->TIME_ARR_DEVICE),
             this->N * sizeof(uint64_t) * this->EXP_IT);
  cudaMalloc(&(this->ADDR_LST_BUF), this->N * sizeof(uint64_t));
  this->TIME_ARR_HOST = new uint64_t[this->N * this->EXP_IT];

  struct cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  this->CLOCK_RATE = device_prop.clockRate;
}

N_Conflict::~N_Conflict()
{
  cudaFree(this->ADDR_LAYOUT);
  cudaFree(&this->TIME_ARR_DEVICE);
  cudaFree(this->ADDR_LST_BUF);
  delete[] this->TIME_ARR_HOST;
}

uint64_t N_Conflict::repeat_n_addr_exp(const uint64_t **addr_arr,
                                       std::ofstream *file)
{
  cudaMemcpy(this->ADDR_LST_BUF, addr_arr, this->N * sizeof(uint64_t),
             cudaMemcpyHostToDevice);

  for (uint64_t i = 0; i < this->EXP_IT; i++)
    n_address_conflict_kernel<<<1, this->N>>>(
        this->ADDR_LST_BUF, this->TIME_ARR_DEVICE + this->N * i);

  cudaDeviceSynchronize();
  cudaMemcpy(this->TIME_ARR_HOST, this->TIME_ARR_DEVICE,
             this->N * sizeof(uint64_t) * this->EXP_IT, cudaMemcpyDeviceToHost);

  uint64_t min = LONG_MAX;
  for (uint64_t i = 0; i < this->EXP_IT; i++)
    min = std::min(this->TIME_ARR_HOST[this->N * i], min);

  if (file)
  {
    *file << '(';
    for (int i = 0; i < this->N; i++)
      *file << *(addr_arr + i) << ", ";
    *file << ")\t" << toNS(min, CLOCK_RATE) << '\n';
  }

  return toNS(min, CLOCK_RATE);
}

} // namespace rbce
