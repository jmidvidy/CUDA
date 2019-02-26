
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

#define BIN_COUNT 1024

__global__ void histogram256Kernel (uint32_t *d_Result, uint32_t *d_Data, int dataN) {

  const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  const int numThreads = blockDim.x * gridDim.x;
  __shared__ uint32_t s_Hist[BIN_COUNT];

  // init bin at thread
  s_Hist[threadIdx.x] = 0;
  d_Result[threadIdx.x] = 0;
  __syncthreads();

  // increment value in shared histogram for value in data at thread
  for (int pos = globalTid; pos < dataN; pos += numThreads) {
    uint data4 = (uint)d_Data[pos];
    atomicAdd(&(s_Hist[data4]), 1);
  }
  
  //merge shared histogram to d_result
  __syncthreads();
  atomicAdd(&(d_Result[threadIdx.x]), s_Hist[threadIdx.x]);
  
}

void opt_2dhisto (uint32_t *d_Result, uint32_t *d_Data, int size_input)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
  dim3 dimBlock(1024,1);
  dim3 dimGrid(16,1);
  histogram256Kernel<<<dimGrid,dimBlock >>>(d_Result, d_Data, size_input);

}

/* Include below the implementation of any other functions you need */

uint32_t* allocateInput(uint32_t *new_input, int size_input) { 

    // put input data into global memory on GPU
    uint32_t *d_Data = new_input;
    cudaMalloc((void**)&d_Data, size_input*sizeof(uint32_t));
    cudaMemcpy(d_Data, new_input, size_input*sizeof(uint32_t),
                                        cudaMemcpyHostToDevice);
    return d_Data;
}

uint32_t* allocateHist(uint32_t *new_bins, int size_hist) { 

    // put final histogram into global memory
    uint32_t *d_Result = new_bins;
    cudaMalloc((void**)&d_Result, size_hist*sizeof(uint32_t));
    cudaMemcpy(d_Result, new_bins, size_hist*sizeof(uint32_t),
                                        cudaMemcpyHostToDevice);
    return d_Result;
}
 
void copyHistToCPU (uint32_t* new_bins, uint32_t* d_Result, int size_hist) {

  //write back to host
  cudaThreadSynchronize();
  cudaMemcpy(new_bins, d_Result, size_hist*sizeof(uint32_t),
                                        cudaMemcpyDeviceToHost);
}

void free_GPU_vars (uint32_t *d_Data ,uint32_t *d_Result){
  cudaFree(d_Data);
  cudaFree(d_Result);
  d_Data = NULL;
  d_Result = NULL;

}
