#include <cub/cub.cuh>
  
     
      // Declare and initialize device pointers for input and output
      int *d_reduce_input, *d_aggregate;
      int num_items = 
     
     
      // Determine temporary device storage requirements for reduction
      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;
      cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_reduce_input, d_aggregate, num_items, cub::Max());
 
  // Allocate temporary storage for reduction
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
 
    // Run reduction (max)
     cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_reduce_input, d_aggregate, num_items, cub::Max(),stream,stream_synchronous);


