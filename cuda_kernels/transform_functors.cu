#pragma once

template<typename T> 
__global__
void to_square_kernel(T* image_in,T* image_out,int heightImage,int widthImage)
{
	unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
 if(x<(heightImage*widthImage))
{
image_out[x]=image_in[x]*image_in[x];
}
}

template <typename T> 
void to_square_transform(T* image_in, T* image_out, int heightImage, int widthImage, int threadsX,cudaStream_t stream)
{	
int nblocks=(heightImage*widthImage)/threadsX;
if((heightImage*widthImage)%threadsX)nblocks++;

	to_square_kernel<<<nblocks,threadsX,0,stream>>>(image_in, image_out, widthImage, heightImage);

}

template<typename T> 
__global__
void multiply_kernel(T* image_in1,T* image_in2,T* image_out,int heightImage,int widthImage)
{
	unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
 if(x<(heightImage*widthImage))
{
image_out[x]=image_in1[x]*image_in2[x];
}
}

template <typename T> 
void multiply_transform(T* image_in1,T*image_in2, T* image_out, int heightImage, int widthImage, int threadsX,cudaStream_t stream)
{	
int nblocks=(heightImage*widthImage)/threadsX;
if((heightImage*widthImage)%threadsX)nblocks++;

	multiply_kernel<<<nblocks,threadsX,0,stream>>>(image_in1,image_in2, image_out, widthImage, heightImage);
}

template<typename T_in,typename T_out> 
__global__
void normalize_kernel(T_in* image_in,T_out* image_out,T_in * max_value,T_out bound_value,int heightImage,int widthImage)
{
	unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
T_in normalizing_constant=(*max_value)/bound_value;
 if(x<(heightImage*widthImage))
{
image_out[x]=(T_out)(image_in[x]/normalizing_constant);
}
}

template <typename T_in,typename T_out> 
void normalize_array(T_in* image_in,T_out* image_out,T_in * max_value,T_out bound_value,int heightImage,int widthImage,const int threadsX,cudaStream_t stream)
{	
int nblocks=(heightImage*widthImage)/threadsX;
if((heightImage*widthImage)%threadsX)nblocks++;

	normalize_kernel<<<nblocks,threadsX,0,stream>>>(image_in,image_out,max_value,bound_value, widthImage, heightImage);
}

template<typename T_in,typename T_out> 
__global__
void normalize_kernel2(T_in* image_in,T_out* image_out,T_in  max_value,T_out bound_value,int heightImage,int widthImage)
{
	unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
T_in normalizing_constant=max_value/bound_value;
 if(x<(heightImage*widthImage))
{
image_out[x]=(T_out)(image_in[x]/normalizing_constant);
}
}

template <typename T_in,typename T_out> 
void normalize_array2(T_in* image_in,T_out* image_out,T_in  max_value,T_out bound_value,int heightImage,int widthImage,const int threadsX,cudaStream_t stream)
{	
int nblocks=(heightImage*widthImage)/threadsX;
if((heightImage*widthImage)%threadsX)nblocks++;

	normalize_kernel2<<<nblocks,threadsX,0,stream>>>(image_in,image_out,max_value,bound_value, widthImage, heightImage);
}