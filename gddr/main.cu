#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cuda.h>
#include <cmath>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CUDA_CHECK_THROW(x)                                                                            \
do {                                                                                                   \
    cudaError_t result = x;                                                                            \
    if (result != cudaSuccess) {                                                                       \
        std::cout << FILE_LINE << " CUDA ERROR: " << cudaGetErrorString(result) << std::endl;          \
        exit(-1);                                                                                      \
    }                                                                                                  \
} while(0);

#define CU_CHECK_THROW(x)                                                                       \
do {                                                                                            \
CUresult result = x;                                                                            \
if (result != CUDA_SUCCESS) {                                                                   \
std::cout << FILE_LINE << " CU ERROR: " << int(result) << std::endl;                            \
exit(-1);                                                                                       \
}                                                                                               \
} while(0);


#define MEM_SPACE (16*1024UL*1024*1024UL)
#define NUM_INT (MEM_SPACE / sizeof(int))
#define NUM_PER_ITER 10000
#define NUM_TOT_ITER 100
#define NUM_EXPERIMENTS 20

__device__ void run_single_experiment(char *__restrict__ ptr,
                                      int stride, float *__restrict__ cyc) {
    for (int i = 0; i < NUM_PER_ITER; ++i) {
        volatile unsigned cyc1 = 0;
        volatile unsigned cyc2 = 0;
        volatile char *target = ptr + (threadIdx.x + 1) * stride;
        __syncwarp(0xFFFFFFFF);
        asm volatile ("{\n\t"
            "mov.u32 %0, %%clock;\n\t"
            "}" :
            "=r"(cyc1)
            :);
        ptr[threadIdx.x] = *target;
        asm volatile ("{\n\t"
            "fence.sc.cta;\n\t"                 // make sure mem ops completed
            "mov.u32 %0, %%clock;\n\t"
            "discard.global.L2 [%1], 128;\n\t"  // flush data from cache
            "}" :
            "=r"(cyc2):
            "l"(target));
        cyc1 = cyc2 - cyc1;
        for (int off = 16; off >= 1; off /= 2) {
            cyc1 += __shfl_down_sync(0xFFFFFFFF, cyc1, off);
        }
        if (threadIdx.x == 0) {
            cyc[i] += cyc1 / 32.0f;
        }
    }
}

// all threads in warp access the same DRAM
__global__ void run_experiment(char *__restrict__ ptr,
                               const int *__restrict__ strides,
                               float **__restrict__ cycles,
                               const unsigned num) {
    for (int i = 0; i < num; ++i) {
        run_single_experiment(ptr, strides[i], cycles[i]);
        __nanosleep(5000);
        __syncwarp(0xffffffff);
    }
}

void print_result(float *dcyc, const int stride_size,
                  std::ostream *file = nullptr) {
    float cyc[NUM_PER_ITER];
    CUDA_CHECK_THROW(
        cudaMemcpy(cyc, dcyc, NUM_PER_ITER * sizeof(float),
            cudaMemcpyDeviceToHost))
    float sum = 0.0f;
    for (int i = 1; i < NUM_PER_ITER; ++i) {
        sum += cyc[i];
    }

    std::cout << stride_size << " " << sum / (NUM_PER_ITER - 1.0f) /
        NUM_TOT_ITER << std::endl;
    if (file != nullptr) {
        *file << stride_size << " " << sum / (NUM_PER_ITER - 1.0f) /
            NUM_TOT_ITER << std::endl;
    }
    CUDA_CHECK_THROW(cudaFree(dcyc))
}

int main() {
    CUdeviceptr dptr;

    CU_CHECK_THROW(cuInit(0))

    size_t granularity = 0;
    CUmemGenericAllocationHandle allocHandle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    CU_CHECK_THROW(cuMemGetAllocationGranularity(&granularity, &prop,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    auto padded_size = MEM_SPACE % granularity
                           ? MEM_SPACE + granularity - (MEM_SPACE % granularity)
                           : MEM_SPACE;
    CU_CHECK_THROW(cuMemCreate(&allocHandle, padded_size, &prop, 0))
    CU_CHECK_THROW(
        cuMemAddressReserve(&dptr, MEM_SPACE, 0, 0x7fee60000000ULL, 0))
    CU_CHECK_THROW(cuMemMap(dptr, MEM_SPACE, 0, allocHandle, 0))

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK_THROW(cuMemSetAccess(dptr, MEM_SPACE, &accessDesc, 1))

    printf("%llx\n", dptr);
    int stride_sizes[NUM_EXPERIMENTS] = {0};
    float stride_latencies[NUM_EXPERIMENTS] = {0.0f};

    int *dstrides;
    float **dcycles;

    CUDA_CHECK_THROW(cudaMalloc(&dstrides, NUM_EXPERIMENTS * sizeof(int)))
    CUDA_CHECK_THROW(cudaMalloc(&dcycles, NUM_EXPERIMENTS * sizeof(float *)))

    for (int i = 0; i < NUM_TOT_ITER; ++i) {
        for (int exp = 0; exp < NUM_EXPERIMENTS; ++exp) {
            float *dcyc;
            CUDA_CHECK_THROW(cudaMalloc(&dcyc, NUM_PER_ITER * sizeof(float)))

            const int stride_size = 128 * pow(2, exp);
            stride_sizes[exp] = stride_size;

            CUDA_CHECK_THROW(cudaMemset(dcyc, 0.0f, NUM_PER_ITER * sizeof(float)))

            CUDA_CHECK_THROW(
                cudaMemcpy(&dcycles[exp], &dcyc, sizeof(float *),
                    cudaMemcpyHostToDevice))
        }

        CUDA_CHECK_THROW(
            cudaMemcpy(dstrides, stride_sizes, NUM_EXPERIMENTS * sizeof(int), cudaMemcpyHostToDevice))
        run_experiment<<<1, 32>>>(reinterpret_cast<char *>(dptr), dstrides, dcycles,
                                  NUM_EXPERIMENTS);
        CUDA_CHECK_THROW(cudaDeviceSynchronize())

        for (int exp = 0; exp < NUM_EXPERIMENTS; ++exp) {
            float *dcyc;
            CUDA_CHECK_THROW(
                cudaMemcpy(&dcyc, &dcycles[exp], sizeof(float *),
                    cudaMemcpyDeviceToHost))

            float cyc[NUM_PER_ITER];
            CUDA_CHECK_THROW(
                cudaMemcpy(cyc, dcyc, NUM_PER_ITER * sizeof(float),
                    cudaMemcpyDeviceToHost))

            for (int k = 1; k < NUM_PER_ITER; ++k) {
                stride_latencies[exp] += cyc[k];
            }

            CUDA_CHECK_THROW(cudaFree(dcyc))
        }
    }


    std::ofstream file("../stride_latencies.txt");

    for (int exp = 0; exp < NUM_EXPERIMENTS; ++exp) {
        std::cout << stride_sizes[exp] << " " << stride_latencies[exp] / ((NUM_PER_ITER - 1) * NUM_TOT_ITER) << std::endl;
        file << stride_sizes[exp] << " " << stride_latencies[exp] / ((NUM_PER_ITER - 1) * NUM_TOT_ITER) << std::endl;
    }

    file.close();
    CU_CHECK_THROW(cuMemUnmap(dptr, MEM_SPACE))
    CU_CHECK_THROW(cuMemRelease(allocHandle))
    CU_CHECK_THROW(cuMemAddressFree(dptr, MEM_SPACE))
    CUDA_CHECK_THROW(cudaFree(dstrides))
    CUDA_CHECK_THROW(cudaFree(dcycles))
    return 0;
}
