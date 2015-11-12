template <typename T> 
	__global__
	void calculate_cornerness_cuda_kernel(T* gx_integral,T* gy_integral,T* gxy_integral,T* cornerness_out,float k_param,int heightImage,int widthImage,int kernel_size)
	{
		unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
		unsigned int y=blockIdx.y*blockDim.y+threadIdx.y;
	
		unsigned int index = x * widthImage + y;
	
		if(y>kernel_size/2+1&&y<widthImage-kernel_size/2&&x>kernel_size/2+1&&x<heightImage-kernel_size/2){
		
			T gxD=gx_integral[(x-kernel_size/2-1)*widthImage+(y-kernel_size/2-1)];
			T gxC=gx_integral[(x+kernel_size/2)*widthImage+(y-kernel_size/2-1)];
			T gxB=gx_integral[(x-kernel_size/2-1)*widthImage+(y+kernel_size/2)];
			T gxA=gx_integral[(x+kernel_size/2)*widthImage+(y+kernel_size/2)];
		
			T sum_gx=gxA+gxD-gxB-gxC;
		
			T gyD=gy_integral[(x-kernel_size/2-1)*widthImage+(y-kernel_size/2-1)];
			T gyC=gy_integral[(x+kernel_size/2)*widthImage+(y-kernel_size/2-1)];
			T gyB=gy_integral[(x-kernel_size/2-1)*widthImage+(y+kernel_size/2)];
			T gyA=gy_integral[(x+kernel_size/2)*widthImage+(y+kernel_size/2)];
		
			T sum_gy=gyA+gyD-gyB-gyC;
		
			T gxyD=gxy_integral[(x-kernel_size/2-1)*widthImage+(y-kernel_size/2-1)];
			T gxyC=gxy_integral[(x+kernel_size/2)*widthImage+(y-kernel_size/2-1)];
			T gxyB=gxy_integral[(x-kernel_size/2-1)*widthImage+(y+kernel_size/2)];
			T gxyA=gxy_integral[(x+kernel_size/2)*widthImage+(y+kernel_size/2)];
		
			T sum_gxy=gxyA+gxyD-gxyB-gxyC;
		
			T det=sum_gx*sum_gy-(sum_gxy*sum_gxy);
			T trace=sum_gx+sum_gy;
		
			cornerness_out[index]=det-k_param*(trace*trace);

			if (cornerness_out[index] < 1 )
				cornerness_out[index] = 0;

		}
		else{cornerness_out[index]=0;}
	  
	  
	}
 
	void calculate_cornerness_cuda( float * gx_integral, float * gy_integral, float * gxy_integral, float * cornerness_out,float k,int mask_size,int heightImage,int widthImage,int cuda_threads ,cudaStream_t stream)
	{
		dim3 block( cuda_threads,cuda_threads, 1);
		dim3 grid( heightImage/ block.x,widthImage / block.y, 1);

		calculate_cornerness_cuda_kernel<<<grid,block,0,stream>>>( gx_integral, gy_integral, gxy_integral, cornerness_out, k, heightImage, widthImage, mask_size);

	}
