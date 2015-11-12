#include "summed_area_table.h"
#include <cub/cub.cuh>

int  scan_horizontally_cudpp( int heightImage, int widthImage, float* input, float* output )
{
	CUDPPConfiguration config = { CUDPP_SCAN, 
		                          CUDPP_ADD, 
		                          CUDPP_FLOAT, 
		                          CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE };

	CUDPPResult ret;

	CUDPPHandle theCudpp;
	ret = cudppCreate(&theCudpp);

    if (ret != CUDPP_SUCCESS)
    {
        std::cerr << "Error Initializing CUDPP Library.\n";
        int retval = 1;
        return retval;
    }

    CUDPPHandle multiscanPlan = 0;
    ret = cudppPlan(theCudpp, &multiscanPlan, config, widthImage, heightImage, widthImage);

    if ( ret != CUDPP_SUCCESS )
    {
        std::cerr << "Error creating CUDPP Plan for multi-row Scan.\n";
        int retval = 1;
        return retval;
    }
	
	ret = cudppMultiScan(multiscanPlan,output,input,widthImage,heightImage);     

	if (ret != CUDPP_SUCCESS)
    {
        std::cerr << "Error in execution of multi-row Scan.\n";
        int retval = 1;
        return retval;
    }
        
   return 0;
}

/**
 * Row scanning kernel.  Each threadblock computes a prefix sum across a row of elements.
 */
template <
    typename BlockScanTilesPolicy,      ///< Tuning policy for cub::BlockScanTiles abstraction
    typename T>                         ///< Data type being scanned
__launch_bounds__ (int(BlockScanTilesPolicy::BLOCK_THREADS))
__global__ void ScanRows(
    T           *d_in,                  ///< Scan input (contiguous rows)
    T           *d_out,                 ///< Scan output (contiguous rows)
    int         row_items)              ///< Number of items per row
{
    // Parameterize the BlockScanTiles type for scanning consecutive "tiles" of input
    typedef cub::BlockScanTiles<
            BlockScanTilesPolicy,       // Tuning policy
            T*,                         // Input iterator type
            T*,                         // Output iterator type
            cub::Sum,                   // Scan functor type
            cub::NullType,              // Identity element type (cub::NullType for inclusive scan)
            int>                        // Integer type for array offsets (e.g., ptr_difft)
        BlockScanTilesT;

    // Shared memory storage
    __shared__ typename BlockScanTilesT::TempStorage temp_storage;

    // Compute begin/end offsets of our block's row
    int block_offset    = blockIdx.x * row_items;
    int block_end       = block_offset + row_items;

    // Instantiate the threadblock instance for tile-scanning
    BlockScanTilesT block_scan_tiles(
        temp_storage,                   // Temporary storage
        d_in,                           // Input
        d_out,                          // Output
        cub::Sum(),                     // Scan functor
        cub::NullType());                    // Identity value

    // Scan the row
    block_scan_tiles.ConsumeTiles(block_offset, block_end);
}

template <typename T>
void ScanRows_wrapper(T* d_in,T* d_out,int num_rows,int row_items,cudaStream_t stream)
{


    // Parameterize a tile-scanning policy
    typedef cub::BlockScanTilesPolicy<
            256,                                ///< Threads per thread block
            2,                                 ///< Items per thread
            cub::BLOCK_LOAD_WARP_TRANSPOSE,     ///< BlockLoadAlgorithm strategy
            false,                              ///< Whether or not only one warp's worth of smem should be used for load-transpositions (and time-sliced)
            cub::LOAD_DEFAULT,                  ///< PtxLoadModifier read modifier (e.g., LOAD_DEFAULT, LOAD_LDG, etc.)
            cub::BLOCK_STORE_WARP_TRANSPOSE,    ///< BlockStoreAlgorithm strategy
            false,                              ///< Whether or not only one warp's worth of smem should be used for store-transpositions (and time-sliced)
            cub::BLOCK_SCAN_RAKING_MEMOIZE>     ///< BlockScanAlgorithm strategy
        BlockScanTilesPolicyT;

    // Run kernel
    ScanRows<BlockScanTilesPolicyT,T><<<num_rows, BlockScanTilesPolicyT::BLOCK_THREADS,0,stream>>>(
        d_in,
        d_out,
        row_items);
}

void scanRows_cub_float(float* d_in,float* d_out,int num_rows,int row_items,cudaStream_t stream)
{
  ScanRows_wrapper<float>(d_in,d_out,num_rows,row_items,stream);
}

void scanRows_new(int num_rows,int row_items,float* d_in,float* d_out,int backend_choice,cudaStream_t stream)
{
  switch(backend_choice)
  {
    case 0:
  scanRows_cub_float(d_in,d_out,num_rows,row_items,stream);
  break;
    case 1:
scan_horizontally_cudpp(num_rows,row_items,d_in,d_out);
  break;

  }
}


