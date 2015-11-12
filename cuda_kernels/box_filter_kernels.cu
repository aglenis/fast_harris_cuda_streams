#pragma once
#include <stdio.h>
#include "transpose_kernel.cu"
#include "summed_area_table.h"

__global__
void compute_grandient_kernel(float* Integral_image,float* Grandient_image,int w,int h,int kernel_size)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = x * w + y;

	if ( ( y >( kernel_size / 2 +1    ) ) &&  
		 ( y < ( w - kernel_size / 2 ) ) && 
		 ( x > ( kernel_size / 2     ) ) &&
		 ( x < ( h - kernel_size / 2 ) ) 
	   ) 
	{
		float B1 = Integral_image[ ( (x - kernel_size / 2) ) * w + (y + kernel_size / 2) ];
		float A1 = Integral_image[ (x + kernel_size / 2 - 1) * w + (y + kernel_size / 2) ];
		float D1 = Integral_image[ (x - kernel_size / 2 + 1-1) * w + (y + 1-1) ];
		float C1 = Integral_image[ (x + kernel_size / 2 - 1) * w + (y + 1-1) ];

		float B2 = Integral_image[ ( (x - kernel_size / 2) + 1-1) * w + (y - 1) ];
		float A2 = Integral_image[ (x + kernel_size / 2 - 1) * w + (y - 1) ];
		float D2 = Integral_image[ (x - kernel_size / 2 + 1-1) * w + (y - kernel_size / 2-1) ];
		float C2 = Integral_image[ (x + kernel_size / 2 - 1) * w + (y - kernel_size / 2-1) ];

		float sum1 = A1 + D1 - B1 - C1;
		float sum2 = A2 + D2 - B2 - C2;

		Grandient_image[index] = (sum1 - sum2) / (kernel_size * kernel_size) ;

	}
	else
	{
		Grandient_image[index] = 0;
	}

}

void callComputeGradient( float* Integral_image, float* Grandient_image, int widthImage, int heightImage, int kernel_size, int threadsX, int threadsY,cudaStream_t stream)
{	
	dim3 block(threadsX, threadsY, 1);
	dim3 grid(heightImage / block.x, widthImage / block.y, 1);

	compute_grandient_kernel<<<grid, block,0,stream>>>( Integral_image, Grandient_image, widthImage, heightImage, kernel_size);
	//cudaThreadSynchronize();

}

void compute_gx(float * integral_in, float * grandient_out, unsigned int h, unsigned int w, int kernel_size, int threadsX, int threadsY,cudaStream_t stream ) 
{
	callComputeGradient( integral_in, grandient_out, w, h, kernel_size, threadsX, threadsY,stream) ;
}

// Last two arguments are temporaries
void compute_gy( float * integral_in, float * grandient_out, unsigned int h, unsigned int w, int kernel_size, int threadsX, int threadsY, float * temp1, float * temp2,cudaStream_t stream) 
{

 	thrust::device_ptr<float> integral_in_thrust( integral_in );
 	thrust::device_ptr<float> grandient_out_thrust( grandient_out );
	thrust::device_ptr<float> temp1_thrust( temp1 );
	thrust::device_ptr<float> temp2_thrust( temp2 );

	transpose(h, w, integral_in_thrust, temp1_thrust );

	int wT = h;
	int hT = w;

	// compute gradient
	callComputeGradient( temp1, temp2, wT, hT, kernel_size, threadsX, threadsY,stream) ;

	// transpose gradient
	transpose( hT, wT, temp2_thrust, grandient_out_thrust ) ;

}


