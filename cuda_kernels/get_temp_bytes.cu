#include <cub/cub.cuh>

size_t get_cub_reduce_temp_bytes(void *d_temp_storage,float * d_reduce_input,float * d_aggregate,int num_items )
{
 size_t temp_storage_bytes = 0;
      cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_reduce_input, d_aggregate, num_items, cub::Max());
return temp_storage_bytes;
}