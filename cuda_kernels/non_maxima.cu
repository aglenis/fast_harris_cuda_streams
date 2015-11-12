#include <stdio.h>

template<typename T> 
__global__
void non_maxima_supression_cuda(T* image_in,T* image_out,int widthImage,int heightImage)
{
	unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int y=blockIdx.y*blockDim.y+threadIdx.y;
	
	unsigned int index = x * widthImage + y;
	if(y>0 && y<(widthImage-1) && x>0 && x<(heightImage-1) ){

		T curr=image_in[index];

		T curr_up=image_in[(x-1)*widthImage+(y)];
		T curr_down=image_in[(x+1)*widthImage+(y)];
		T curr_up_left=image_in[(x-1)*widthImage+(y-1)];
		T curr_up_right=image_in[(x-1)*widthImage+(y+1)];

		T curr_down_left=image_in[(x+1)*widthImage+(y-1)];
		T curr_down_right=image_in[(x+1)*widthImage+(y+1)];

		T curr_left=image_in[(x)*widthImage+(y-1)];
		T curr_right=image_in[(x)*widthImage+(y+1)];

		T max_element=curr;

		if(curr_up>max_element)max_element=curr_up;
		if(curr_down>max_element)max_element=curr_down;

		if(curr_down_left>max_element)max_element=curr_down_left;
		if(curr_down_right>max_element)max_element=curr_down_right;

		if(curr_up_left>max_element)max_element=curr_up_left;
		if(curr_up_right>max_element)max_element=curr_up_right;

		if(curr_left>max_element)max_element=curr_left;
		if(curr_right>max_element)max_element=curr_right;

		if ( max_element != curr
		  ||max_element==curr_up||max_element==curr_up_left||max_element==curr_up_right||max_element==curr_left )
			image_out[ index ] = 0;
		else
			image_out[ index ] = curr;

	}
	else
	{
		image_out[index]=0;
	}
}


template <typename T> 
void calculate_non_maxima_cuda(T* image_in, T* image_out, int heightImage, int widthImage, int threadsX, int threadsY,cudaStream_t stream)
{	

	dim3 block(threadsX, threadsY, 1);
	dim3 grid(heightImage / block.x, widthImage / block.y, 1);

	non_maxima_supression_cuda<<<grid,block,0,stream>>>(image_in, image_out, widthImage, heightImage);

}
