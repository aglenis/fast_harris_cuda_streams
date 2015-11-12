#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>


template <typename T, int TD_T, int BLOCK_ROWS>
__global__ void transposeNoBankConflicts(T *odata, T *idata, int width, int height)
{
	__shared__ T tile[TD_T][TD_T+1];

	int xIndex = blockIdx.x * TD_T + threadIdx.x;
	int yIndex = blockIdx.y * TD_T + threadIdx.y;  

	int index_in = xIndex + yIndex * width;

	xIndex = blockIdx.y * TD_T + threadIdx.x;
	yIndex = blockIdx.x * TD_T + threadIdx.y;

	int index_out = xIndex + (yIndex) * height;

	if ( xIndex < height && yIndex < width)
	{
		for (int i=0; i<TD_T; i+=BLOCK_ROWS) 
		{
			tile[threadIdx.y + i][threadIdx.x] = idata[ index_in + i * width];
		}	
 
		__syncthreads();

		for ( int i = 0; i < TD_T; i += BLOCK_ROWS) 
		{
			odata[ index_out + i * height] = tile[ threadIdx.x ][ threadIdx.y + i ];
		}
	}
}

template <typename T>
void transpose_2( size_t height, size_t width, T* idata, T* odata, int tile_dimension,cudaStream_t stream )
{
	int gridx=width / tile_dimension;
	if	( width % tile_dimension != 0)
	{
		++ gridx;
		std::stringstream ss;
		ss << "Transpose 2: Width " << width << " is not divisible by tile dimension: " << tile_dimension << ". Aborting\n";
		throw std::runtime_error( ss.str() );
	}
  
	int gridy = height/tile_dimension;
  	if ( height % tile_dimension != 0)
	{
		++ gridy;
		std::stringstream ss;
		ss << "Transpose 2: Height " << height<< " is not divisible by tile dimension: " << tile_dimension << ". Aborting\n";
		throw std::runtime_error( ss.str() );
	}
  
	dim3 grid( gridx, gridy), 
		threads( tile_dimension, tile_dimension );
	switch (tile_dimension)
	{
		case 2:
			transposeNoBankConflicts<T,2,2><<<grid, threads,0,stream>>>(odata, idata, width, height);
			break;
		case 4:
			transposeNoBankConflicts<T,4,4><<<grid, threads,0,stream>>>(odata, idata, width, height);
		break;
		case 8:
			transposeNoBankConflicts<T,8,8><<<grid, threads,0,stream>>>(odata, idata, width, height);
		break;
		case 16:
			transposeNoBankConflicts<T,16,16><<<grid, threads,0,stream>>>(odata, idata, width, height);
		break;
/*		case 24:
			transposeNoBankConflicts<T,24,24><<<grid, threads>>>(odata, idata, width, height);
			break;
		case 32:
			transposeNoBankConflicts<T,32,32><<<grid, threads>>>(odata, idata, width, height);
			break;
		case 64:
			transposeNoBankConflicts<T,64,64><<<grid, threads>>>(odata, idata, width, height);
		case 128:
			transposeNoBankConflicts<T,128,128><<<grid, threads>>>(odata, idata, width, height);
*/		default:
			std::cerr << "Tile Dimension: " << tile_dimension << " not supported. Aborting\n";
			exit( -1 );
	}
}

