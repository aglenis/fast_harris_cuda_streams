#pragma once
#include <thrust/scan.h>

#include "summed_area_table.h"
#include "transpose_kernel.cu"
#include "transform_functors.cu"

template<typename T>
struct to_square_thrust : public thrust::unary_function<T,T>
{
	__host__ __device__
	T operator()(T x) const { return x*x;	}
};

template<typename T>
void compute_integral_image_new( T * input, T* output, int numRows, int numCols, int tile_dimension, T* temp1,int backend_choice,cudaStream_t stream )
{
 	thrust::device_ptr<T> input_thrust( input );
 	thrust::device_ptr<T> output_thrust( output );
	thrust::device_ptr<T> temp1_thrust( temp1 );

	scanRows_new( numRows, numCols, input, temp1,backend_choice,stream );
	transpose_2(numRows, numCols, temp1, output, tile_dimension,stream );
	scanRows_new( numCols, numRows, output, temp1,backend_choice,stream ) ;
	transpose_2(numCols, numRows, temp1, output, tile_dimension,stream );
	
}

template<typename T>
void compute_transposed_integral_image_new( T * input, T* output, int numRows, int numCols, int tile_dimension, T* temp1,int backend_choice,cudaStream_t stream )
{
 	thrust::device_ptr<T> input_thrust( input );
 	thrust::device_ptr<T> output_thrust( output );
	thrust::device_ptr<T> temp1_thrust( temp1 );

	scanRows_new( numRows, numCols, input, output,backend_choice,stream );
	transpose_2(numRows, numCols, output, temp1, tile_dimension,stream );
	scanRows_new( numCols, numRows, temp1, output,backend_choice,stream ) ;
}

template<typename T1, typename T2>
void compute_integral_image( T1 * input, T2* output, int numRows, int numCols, T2* temp1, int tile_dimension = 2 )
{
 	thrust::device_ptr<T1> input_thrust( input );
 	thrust::device_ptr<T2> output_thrust( output );
	thrust::device_ptr<T2> temp1_thrust( temp1 );

	scan_horizontally( numRows, numCols, input_thrust, temp1_thrust ) ;
	transpose_2(numRows, numCols, temp1, output, tile_dimension );
	scan_horizontally( numCols, numRows, output_thrust, temp1_thrust ) ;
	transpose_2(numCols, numRows, temp1, output, tile_dimension );

}

template<typename T>
void compute_integral_image( T * input, T* output, int numRows, int numCols, T* temp1, int tile_dimension = 2 )
{
 	thrust::device_ptr<T> input_thrust( input );
 	thrust::device_ptr<T> output_thrust( output );
	thrust::device_ptr<T> temp1_thrust( temp1 );

	scan_horizontally( numRows, numCols, input_thrust, temp1_thrust ) ;
	transpose_2(numRows, numCols, temp1, output, tile_dimension );
	scan_horizontally( numCols, numRows, output_thrust, temp1_thrust ) ;
	transpose_2(numCols, numRows, temp1, output, tile_dimension );

}

template<typename T>
void compute_squared_integral_image( T* input, T* output, int numRows, int numCols, int tile_dimension, T* temp1 )
{
 	thrust::device_ptr<T> input_thrust( input );
 	thrust::device_ptr<T> output_thrust( output );
	thrust::device_ptr<T> temp1_thrust( temp1 );

	scan_horizontally( numRows, numCols, thrust::make_transform_iterator(input_thrust, to_square_thrust<T>() ), temp1_thrust ) ;
	transpose_2(numRows, numCols, temp1, output, tile_dimension );
	scan_horizontally( numCols, numRows, output_thrust, temp1_thrust ) ;
	transpose_2(numCols, numRows, temp1, output, tile_dimension );
}


template<typename T>
void compute_squared_integral_image_new( T* input, T* output, int numRows, int numCols, T* temp1,int backend_choice,cudaStream_t stream )
{
if(backend_choice==0)
{
const int cuda_threads_x_only=256;
to_square_transform(input,output, numRows,numCols,cuda_threads_x_only,stream);
}
else
{
 	thrust::device_ptr<T> input_thrust( input );
 	thrust::device_ptr<T> output_thrust( output );
	thrust::device_ptr<T> temp1_thrust( temp1 );

	thrust::transform( input_thrust, input_thrust + numRows * numCols, output_thrust, to_square_thrust<T>() );
}
	compute_integral_image_new( output, output, numRows, numCols, temp1,backend_choice,stream);
}


template<typename T>
void compute_transposed_squared_integral_image_new( T* input, T* output, int numRows, int numCols, int tile_dimension, T* temp1,int backend_choice,cudaStream_t stream )
{
if(backend_choice==0)
{
const int cuda_threads_x_only=256;
to_square_transform(input,output, numRows,numCols,cuda_threads_x_only,stream);
}
else
{
 	thrust::device_ptr<T> input_thrust( input );
 	thrust::device_ptr<T> output_thrust( output );
	thrust::device_ptr<T> temp1_thrust( temp1 );

	thrust::transform( input_thrust, input_thrust + numRows * numCols, output_thrust, to_square_thrust<T>() );
}
	compute_transposed_integral_image_new( output, output, numRows, numCols, tile_dimension, temp1,backend_choice,stream);
}


template<typename T>
void compute_multiplied_integral_image(T* input1, T* input2, T* output, int numRows, int numCols, T* temp1, T* temp2)
{
 	thrust::device_ptr<T> input2_thrust( input1 );
 	thrust::device_ptr<T> input1_thrust( input2 );
 	thrust::device_ptr<T> output_thrust( output );
	thrust::device_ptr<T> temp1_thrust( temp1 );
	thrust::device_ptr<T> temp2_thrust( temp2 );

	thrust::transform( input1_thrust, input1_thrust + numRows * numCols, input2_thrust, output_thrust, thrust::multiplies<T>() );

	compute_integral_image( output, output, numRows, numCols, temp1);
}

template<typename T>
void compute_multiplied_integral_image_new(T* input1, T* input2, T* output, int numRows, int numCols, T* temp1,int backend_choice,cudaStream_t stream)
{
if(backend_choice==0)
{
const int cuda_threads_x_only=256;
multiply_transform(input1,input2,output, numRows,numCols,cuda_threads_x_only,stream);
}
else
{
 	thrust::device_ptr<T> input2_thrust( input1 );
 	thrust::device_ptr<T> input1_thrust( input2 );
 	thrust::device_ptr<T> output_thrust( output );
	thrust::device_ptr<T> temp1_thrust( temp1 );

	thrust::transform( input1_thrust, input1_thrust + numRows * numCols, input2_thrust, output_thrust, thrust::multiplies<T>() );
}
	compute_integral_image_new( output, output, numRows, numCols, temp1,backend_choice,stream);
}

template<typename T>
void compute_transposed_multiplied_integral_image_new(T* input1, T* input2, T* output, int numRows, int numCols, int tile_dimension, T* temp1,int backend_choice,cudaStream_t stream )
{
if(backend_choice==0)
{
const int cuda_threads_x_only=256;
multiply_transform(input1,input2,output, numRows,numCols,cuda_threads_x_only,stream);
}
else
{
 	thrust::device_ptr<T> input2_thrust( input1 );
 	thrust::device_ptr<T> input1_thrust( input2 );
 	thrust::device_ptr<T> output_thrust( output );
	thrust::device_ptr<T> temp1_thrust( temp1 );

	thrust::transform( input1_thrust, input1_thrust + numRows * numCols, input2_thrust, output_thrust, thrust::multiplies<T>() );
}
	compute_transposed_integral_image_new( output, output, numRows, numCols, tile_dimension, temp1,backend_choice,stream);
}

