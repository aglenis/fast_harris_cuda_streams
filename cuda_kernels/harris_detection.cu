#include "box_filter_kernels.cu"
#include "calculate_cornerness_kernel.cu"
#include "kernels_integral.cu"
#include "non_maxima.cu"
#include "transform_functors.cu"
#include "castor.h"
#include <exception>
#include <iostream>

void check_for_errors( const std::string & function_call )
	{
		cudaError e = cudaGetLastError();
		if ( e != cudaSuccess )
			std::cout<<cudaGetErrorString(e)<<std::endl;
		// else std::cerr << function_call << "--- DONE\n";
	}
void harris_features_detection( float * gpu_image_in, uint8_t * image_out,float * workspace1,float * workspace2,float * workspace3,float * workspace4,void *d_temp_storage,size_t temp_storage_bytes,float * d_aggregate,int heightImage,int widthImage,cudaStream_t stream) 
	{
		int sz =  heightImage*widthImage;

		int widthImageT = heightImage;
		int heightImageT = widthImage;
		
		const int tile_dimension = 16;
		const int cuda_threads=tile_dimension;
		const int cuda_threads_x_only=256;
		const int mask_size=9;
		const float k=0.01;
		const int backend_choice=0;
		
		bool use_thrust_normalize=0;


		// 1. calculate intergral image. also save its transposed - it will also be used directly 
		float * integral_image = workspace1; 
		float * integral_image_tr = workspace2; 
		compute_integral_image_new( gpu_image_in, integral_image, heightImage, widthImage, tile_dimension, integral_image_tr,backend_choice,stream );
		check_for_errors( "compute_integral_image_new");
 
		// 2. calculate the x derivate of the integral image and transpose it. the intergral image is no further used 
		float * gx = workspace3;
		float * gx_tr = workspace1;
		compute_gx( integral_image, gx, heightImage, widthImage, mask_size, cuda_threads, cuda_threads,stream); 
		check_for_errors( "compute_gx");
		transpose_2( heightImage, widthImage, gx, gx_tr, tile_dimension,stream ) ;
		check_for_errors( "transpose_2");

		// 2. calculate the y derivate of the transposed integral image. the transposed intergral image is no further used
		float * gy_tr = workspace3;
		compute_gx( integral_image_tr, gy_tr, widthImage, heightImage, mask_size, cuda_threads, cuda_threads,stream );
		check_for_errors( "compute_gx");

		// 3. calculate the integral image of the transposed squared x derivative of the integral image
		float * temp1 = workspace4;
		float * integral_gx_tr = workspace2;
		compute_transposed_squared_integral_image_new(gx_tr, integral_gx_tr, heightImageT, widthImageT, tile_dimension, temp1,backend_choice,stream );
		check_for_errors( "compute_transposed_squared_integral_image_new");
 

		// 4. calculate the produce ot the transposed x and y derivatives of the integral image. gx_tr is no longer needed
		float * integral_gxgy_tr = gx_tr; 
		compute_transposed_multiplied_integral_image_new(gx_tr, gy_tr, integral_gxgy_tr, heightImageT, widthImageT, tile_dimension, temp1,backend_choice,stream );
		check_for_errors("compute_transposed_multiplied_integral_image_new" );

		// 5. calculate the integral image of the transposed squared y derivative of the integral image. gy_tr is no longer needed
		float * integral_gy_tr = gy_tr;
		compute_transposed_squared_integral_image_new(gy_tr, integral_gy_tr, heightImageT, widthImageT, tile_dimension, temp1,backend_choice,stream );
		check_for_errors("compute_transposed_squared_integral_image_new" );

		// 6. calculate cornerness based on intergral images of the derivatives 
		float * cornerness = workspace4;
		calculate_cornerness_cuda(integral_gx_tr, integral_gy_tr, integral_gxgy_tr, cornerness,k,mask_size,heightImage,widthImage,cuda_threads,stream);
		check_for_errors("calculate_cornerness_cuda" );

		// 7. remove non maximal cornerness values 
		float * gpu_image_out = workspace1;
		calculate_non_maxima_cuda( cornerness, gpu_image_out , heightImage, widthImage, cuda_threads, cuda_threads,stream);	
		check_for_errors("calculate_non_maxima_cuda" );

		//8. normalise values from 0 to 255
if(use_thrust_normalize){
 		thrust::device_ptr<float> result_begin( cornerness );
 		float maxvalue = thrust::reduce( result_begin, result_begin + sz, 0.0f, thrust::maximum<float>() );
// 		thrust::transform( result_begin, result_begin + sz, thrust::device_ptr<uint8_t>( image_out ), normalise_functor<float,uint8_t>( maxvalue / 255) );
 normalize_array2<float,uint8_t>(cornerness,image_out,maxvalue,255,heightImage,widthImage,cuda_threads_x_only,stream);
}
else
{    
float *d_reduce_input=cornerness;
bool stream_synchronous=0;
int num_items=sz;
 
    // Run reduction (max)
     cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_reduce_input, d_aggregate, num_items, cub::Max(),stream,stream_synchronous);

 normalize_array<float,uint8_t>(cornerness,image_out,d_aggregate,255,heightImage,widthImage,cuda_threads_x_only,stream);
//   cudaFree(d_temp_storage);
 }
		check_for_errors("normalisor" );

		}