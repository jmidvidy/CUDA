#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 512
#define SECTION_SIZE 1024

// Lab4: Host Helper Functions (allocate your own data structure...)

// Lab4: Device Functions

__global__ void prefixIncr(float *incrs, float *nums, int size) {
	int globalTid = blockIdx.x*blockDim.x + threadIdx.x;
	int i = globalTid;
	int numThreads = blockDim.x*gridDim.x;	
	
	int count = 0;
	while (i < size) {
		nums[i] += incrs[blockIdx.x + gridDim.x*count];
		count += 1;
		i += numThreads;
	}
}


__global__ void prescan2(float *X, float *Y, int paddedSize, int origSize, float *SUMS) {
	__shared__ float XY[SECTION_SIZE];
	int i = 2*blockIdx.x*blockDim.x+threadIdx.x;
	
	int count = 0;

	while (i < paddedSize) {
		if (i == 0 || i >= origSize)
			XY[threadIdx.x] = 0;
		else {
			XY[threadIdx.x] = X[i-1];
		}

		if (i + blockDim.x < paddedSize) {
			if (i + blockDim.x < origSize)
				XY[threadIdx.x + blockDim.x] = X[i + blockDim.x-1];
			else
				XY[threadIdx.x + blockDim.x] = 0;
		}
		
		// reduction part
		for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
			__syncthreads();
			int index = (threadIdx.x+1)*2*stride - 1;
			if (index < SECTION_SIZE) {
				XY[index] += XY[index - stride];
			}
		}

		// distribution part
		for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
			__syncthreads();
			int index = (threadIdx.x + 1)*stride*2 - 1;
			if (index + stride < SECTION_SIZE) {
			  XY[index + stride] += XY[index];
			}
		}

		// combining
		__syncthreads();
		
		if (i < origSize) Y[i] = XY[threadIdx.x];
		if (i + blockDim.x < origSize) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
		if (threadIdx.x == 0) SUMS[blockIdx.x] = XY[SECTION_SIZE-1];	
		i += SECTION_SIZE*gridDim.x;
		count++;
	}
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{
	int pwr = ceil(log2((double)numElements));
	int n = pow(2, pwr);
	float *SUMS;
	float *SUMS2;
	float *INCR;
	float *INCR2;
	const int numBlocks = n/1024;
	const int numBlocks2 = numBlocks/1024;
	cudaMalloc( (void**) &SUMS, numBlocks*sizeof(float));
	cudaMalloc( (void**) &INCR, numBlocks*sizeof(float));
	cudaMalloc( (void**) &INCR2, numBlocks2*sizeof(float));
	
	dim3 dimBlock(512,1);
	dim3 dimGrid(numBlocks,1);
	prescan2<<<dimGrid,dimBlock >>>(inArray, outArray, n, numElements, SUMS);

	if (numBlocks2 > 0) {

		cudaMalloc( (void**) &SUMS2, numBlocks2*sizeof(float));
		
		// always one block?? seems like we may have too many threads?
		dim3 dimBlock_2(512,1);
		dim3 dimGrid_2(numBlocks2,1);
		prescan2<<<dimBlock_2,dimBlock_2 >>>(SUMS, INCR, numBlocks, numBlocks, SUMS2);
		
		float *SUMS2_COPY = new float[numBlocks2];
		cudaMemcpy(SUMS2_COPY, SUMS2, sizeof(float) * numBlocks2, cudaMemcpyDeviceToHost);
	
		float *incr2 = (float*)malloc(sizeof(float)*numBlocks2);

		incr2[0] = 0;
		for (int i = 0; i < numBlocks2-1; i++) {
			incr2[i+1] = SUMS2_COPY[i] + incr2[i];
		}
		
		cudaMemcpy(INCR2, incr2, sizeof(float) * numBlocks2, cudaMemcpyHostToDevice);
		
		dim3 dimBlock_3(1024,1);
		dim3 dimGrid_3(16,1);
		prefixIncr<<<dimBlock_3,dimBlock_3 >>>(INCR2, INCR, numBlocks);

		free(incr2);
		free(SUMS2_COPY);
	}
	else {
		dim3 dimBlock_2(512,1);
		dim3 dimGrid_2(16,1);
		cudaMalloc( (void**) &SUMS2, sizeof(float));
		prescan2<<<dimBlock_2,dimBlock_2 >>>(SUMS, INCR, numBlocks, numBlocks, SUMS2);
	}
	
	prefixIncr<<<numElements/1024,1024>>>(INCR, outArray, numElements);

	cudaFree(SUMS);
	cudaFree(INCR);
	cudaFree(INCR2);
	cudaFree(SUMS2);	
}


// **===-----------------------------------------------------------===**



#endif // _PRESCAN_CU_
