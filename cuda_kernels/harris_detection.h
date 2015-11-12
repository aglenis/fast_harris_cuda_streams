#include<cuda.h>
#include<cuda_runtime.h>
 #include "helper_cuda.h"
void harris_features_detection( float * gpu_image_in, uint8_t * image_out,float * workspace1,float * workspace2,float * workspace3,float * workspace4,void *d_temp_storage,size_t temp_storage_bytes,float * d_aggregate,int heightImage,int widthImage,cudaStream_t stream); 