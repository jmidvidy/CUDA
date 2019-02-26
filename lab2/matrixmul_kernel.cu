/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
#define TILE_WIDTH 32

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix d_M, Matrix d_N, Matrix d_P)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH+1];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH+1];

	printf("IN KERNEL!!\n");
	
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;
	for (int ph = 0; ph < ceil(d_P.width/(float)TILE_WIDTH); ++ph) {
	  if ((Row < d_M.height) && (ph*TILE_WIDTH+tx) < d_M.width)
		Mds[ty][tx] = d_M.elements[Row*d_M.width + ph*TILE_WIDTH + tx];
	  else
		Mds[ty][tx] = 0;	  
	  if ((ph*TILE_WIDTH + ty) < d_N.height && Col < d_N.width)
		Nds[ty][tx] = d_N.elements[(ph*TILE_WIDTH + ty)*d_N.width + Col];
	  else
		Nds[ty][tx] = 0;
	  __syncthreads();
	
  	  for (int k = 0; k < TILE_WIDTH; ++k) {
		Pvalue += Mds[ty][k] * Nds[k][tx];
	  }
	  __syncthreads();
	}
	//printf("\n%f", Pvalue);
	if ((Row<d_P.height) && (Col < d_P.width))
	  d_P.elements[Row*d_P.width +  Col] = Pvalue;
}

#endif // #ifnde _MATRIXMUL_KERNEL_H_
