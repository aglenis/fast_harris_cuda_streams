#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include "cuda_kernels/harris_detection.h"
#include "../timer.h"
#include "get_temp_bytes.h"

int main(int argc, char** argv)
{
	int key = -1;
	float fTotalTime = 0;
	unsigned int iFrameCnt, iFrameStart;
	iFrameStart = iFrameCnt = 0;
	
	CvCapture* capture = 0;
	capture = cvCaptureFromAVI(argv[1]);
	int TARGET_WIDTH=atoi(argv[2]);
	int TARGET_HEIGHT=atoi(argv[3]);
	bool visualize_results=atoi(argv[4]);
	int num_streams=atoi(argv[5]);
	int total_buffer_frames=num_streams;
	if( !capture )
	{
		fprintf(stderr,"Could not initialize...\n");
		return -1;
	}

	IplImage ** array_of_frames;
	
 	array_of_frames=(IplImage **)malloc(total_buffer_frames*sizeof(IplImage *));
	IplImage* videoFrame = NULL;
	videoFrame = cvQueryFrame(capture);
	
	IplImage ** array_of_gray_frames;
 	array_of_gray_frames=(IplImage **)malloc(total_buffer_frames*sizeof(IplImage *));
	
	IplImage ** array_of_resized_gray_frames;
 	array_of_resized_gray_frames=(IplImage **)malloc(total_buffer_frames*sizeof(IplImage *));
	IplImage ** array_of_output_frames;
 	array_of_output_frames=(IplImage **)malloc(total_buffer_frames*sizeof(IplImage *));
	
	
	for( int i=0;i<total_buffer_frames;i++){
	array_of_gray_frames[i] = cvCreateImage(cvSize(videoFrame->width, videoFrame->height), IPL_DEPTH_8U, 1);
	array_of_resized_gray_frames[i] = cvCreateImage(cvSize(TARGET_WIDTH,TARGET_HEIGHT), IPL_DEPTH_8U, 1);
	array_of_output_frames[i] = cvCreateImage(cvSize(array_of_resized_gray_frames[i]->width, array_of_resized_gray_frames[i]->height), IPL_DEPTH_8U, 1);
	}
	
	int sz_float=array_of_resized_gray_frames[0]->height*array_of_resized_gray_frames[0]->width*sizeof(float);
	int sz=array_of_resized_gray_frames[0]->height*array_of_resized_gray_frames[0]->width;
	float **d_workspace1,**d_workspace2,**d_workspace3,**d_workspace4,**d_input_image;
	float **d_aggregate;
	d_workspace1=(float **)malloc(num_streams*sizeof(float*));
	d_workspace2=(float **)malloc(num_streams*sizeof(float*));
	d_workspace3=(float **)malloc(num_streams*sizeof(float*));
	d_workspace4=(float **)malloc(num_streams*sizeof(float*));
	d_input_image=(float **)malloc(num_streams*sizeof(float*));
	d_aggregate=(float **)malloc(num_streams*sizeof(float*));
	unsigned char **d_output_image;
	d_output_image=(unsigned char **)malloc(num_streams*sizeof(unsigned char*));
	for( int i=0;i<num_streams;i++){
	
	checkCudaErrors(cudaMalloc(&d_workspace1[i],sz_float));
	checkCudaErrors(cudaMalloc(&d_workspace2[i],sz_float));
	checkCudaErrors(cudaMalloc(&d_workspace3[i],sz_float));
	checkCudaErrors(cudaMalloc(&d_workspace4[i],sz_float));
	checkCudaErrors(cudaMalloc(&d_input_image[i],sz_float));
	checkCudaErrors(cudaMalloc(&d_output_image[i],sz*sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc(&d_aggregate[i],sizeof(float)));
	}
	cudaStream_t * streams_array;
	streams_array=(cudaStream_t*)malloc(num_streams*sizeof(cudaStream_t));
	
	for( int i=0;i<num_streams;i++){
	checkCudaErrors(cudaStreamCreate(&(streams_array[i])));
	}
	
	
      int num_items =sz;
     bool stream_synchronous=1;
           // Determine temporary device storage requirements for reduction
      void **d_temp_storage;
      d_temp_storage=(void**)malloc(num_streams*sizeof(void *));
      d_temp_storage[0]=NULL;
      float * d_reduce_input=NULL;
     size_t temp_storage_bytes=get_cub_reduce_temp_bytes(d_temp_storage[0],d_reduce_input,d_aggregate[0],num_items );

  // Allocate temporary storage for reduction

     for (int i=0;i<num_streams;i++){
      checkCudaErrors(cudaMalloc(&d_temp_storage[i], temp_storage_bytes));
     }
	float ** input_frame;
	input_frame=(float **)malloc(num_streams*sizeof(float*));
	for (int i=0;i<num_streams;i++)
	{
	input_frame[i]=(float *)malloc(sz_float);
	}
	if(!videoFrame)
	{
		printf("Bad frame \n");
		exit(0);
	}

	cvNamedWindow("input", 1);
	cvNamedWindow("gray", 1);
	cvNamedWindow("resized", 1);
	cvNamedWindow("output", 1);
	
	double total_time_elapsed=0.0f;
	double total_time_elapsed_mem=0.0f;
	unsigned total_frames=0;
		while(key != 'q')
	{
	  bool have_to_break=false;
	  for( int i=0;i<num_streams;i++)
	  {videoFrame = cvQueryFrame(capture);

		if( !videoFrame){
			have_to_break=true;
			break;
		}

		cvCvtColor(videoFrame, array_of_gray_frames[i], CV_BGR2GRAY);
		cvResize(array_of_gray_frames[i],array_of_resized_gray_frames[i]);
		for( int i=0;i<num_streams;i++){
		for (int j=0;j<sz;j++)
		{float curr_value=(float)((array_of_resized_gray_frames[i])->imageData[j]);
		  input_frame[i][j]=curr_value;}
		iFrameCnt++;
		}
	  }
	  if(have_to_break)break;
		double fEllapsed;
		
		timer timer_mem1;
		for( int i=0;i<num_streams;i++){
		checkCudaErrors(cudaMemcpyAsync((void*)d_input_image[i],(void*)input_frame[i],sz_float,cudaMemcpyHostToDevice,streams_array[i]));}
		total_time_elapsed_mem+=timer_mem1.elapsed();
		timer time_harris;
		for( int i=0;i<num_streams;i++){
		harris_features_detection(d_input_image[i], d_output_image[i],d_workspace1[i],d_workspace2[i],d_workspace3[i],d_workspace4[i],d_temp_storage[i],temp_storage_bytes,d_aggregate[i],array_of_resized_gray_frames[0]->height,array_of_resized_gray_frames[0]->width,streams_array[i]); 
		}
		double total_time_harris=time_harris.elapsed();
		total_time_elapsed+=total_time_harris;
		total_time_elapsed_mem+=total_time_harris;
		timer timer_mem2;
		for( int i=0;i<num_streams;i++){
		checkCudaErrors(cudaMemcpyAsync((void*)array_of_output_frames[i]->imageData,(void*)d_output_image[i],sz*sizeof(unsigned char),cudaMemcpyDeviceToHost,streams_array[i]));
		}
		for( int i=0;i<num_streams;i++){
		checkCudaErrors(cudaStreamSynchronize	(streams_array[i]));
		}

		total_time_elapsed_mem+=timer_mem2.elapsed();
		total_frames++;
		if(visualize_results)
		{
		for(int i=0;i<total_buffer_frames;i++){  
		cvShowImage("input", videoFrame);
		cvShowImage("gray", array_of_gray_frames[i]);
		cvShowImage("resized", array_of_resized_gray_frames[i]);
		cvShowImage("output",array_of_output_frames[i]);
		key=cv::waitKey(10);
		}
		}
// 		return 0;
	}
	printf("Total Frames per seconds were %f \n",total_frames/total_time_elapsed);
	printf("Total Frames per seconds with memcopies were %f \n",total_frames/total_time_elapsed_mem);
}