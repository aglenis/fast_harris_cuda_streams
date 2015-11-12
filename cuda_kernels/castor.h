#pragma once

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>


	template< typename T1, typename T2>
	struct cast_functor : public thrust::unary_function<T1,T2>
	{
		typedef T1 in_t;
		typedef T2 out_t;
		

		__host__ __device__ out_t operator()(const in_t & x) const
		{ 
			return out_t( x );
		}
	
	};	    

	template< typename T1, typename T2>
	struct normalise_functor : public thrust::unary_function<T1,T2>
	{
		typedef T1 in_t;
		typedef T2 out_t;

		T1 normalisation_constant;
		
		normalise_functor( T1 _normalisation_constant )
			: normalisation_constant( _normalisation_constant)
		{}
		
		__host__ __device__
		out_t operator()(const in_t & x) const  
		{ 
			return static_cast<out_t>( ( x / normalisation_constant ) );
		}
	
	};	    

	struct uchar4ToFloat: public thrust::unary_function<uchar4,float>
	{
		__host__ __device__
		float operator()(const uchar4 & in_element) const  
		{ 
			
			return (float) (in_element.x + in_element.y + in_element.z) / 3.0 ;
		}
	
	};	 



