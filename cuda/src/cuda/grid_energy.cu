#include "grid.cuh"

// NOTE: Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-parameters
//
//     __global__ function parameters are PASSED TO THE DEVICE via constant memory and are LIMITED TO 4 KB.
//     __global__ functions cannot have a variable number of arguments.
//     __global__ function parameters CANNOT BE PASS-BY-REFERENCE.
//
// Since there is a limit of 4KB (i.e., 512 doubles), let's make a struct that contains only data members.



template<int N_block, int N_thread>
double cuda_Class_Grid::energy_calculation ()
{
    auto & T = this->thrust_memory;

    // if ( (G_type_x - 'M') + (G_type_y - 'M') == ('N' - 'M') * 2 )  // NN grid
    if ( strcmp( grid_name.c_str(), "SMM" ) == 0 )
    {
        auto & Sxx = Vec_soln .at(0);
        auto & Syy = Vec_soln .at(1);

        auto & P1  = Vec_prmt_enrg.at(0);
        auto & P2  = Vec_prmt_enrg.at(1);

        weighted_square_NORMAL_grid <<< N_block , N_thread >>> ( this->struct_grid , 
                                                                 Sxx.ptr , Syy.ptr , 
                                                                 P1 .ptr , P2 .ptr , 
                                                                 T  .ptr );  // T needs to be allocated thrust_memory ?
    }
    else
    {
        auto & S = Vec_soln .at(0);
        auto & P = Vec_prmt_enrg.at(0);        
        auto & T = this->thrust_memory;

        weighted_square_SINGLE_grid <<< N_block , N_thread >>> ( this->struct_grid , 
                                                                 S.ptr , P.ptr , T.ptr );  // T needs to be allocated thrust_memory ?
    }

    thrust::device_ptr<double> d_ptr ( T.ptr );
    double E = thrust::reduce( d_ptr, d_ptr + T.length );

    return E/2.;
}



__global__ void weighted_square_NORMAL_grid ( cuda_Struct_Grid struct_grid , 
                                              ns_type::cuda_precision * Sxx , ns_type::cuda_precision * Syy , 
                                              double * P1  , double * P2  , 
                                              double * T )  // T stores the intermediate output 
{    
    // int N_block  =  gridDim.x;    // Not needed because we don't loop through the (simulation) gridLines (one threadblock each gridLine)
    int N_thread = blockDim.x;

    int ind_block  =  blockIdx.x;
    int ind_thread = threadIdx.x;

    int ix = ind_block;
    if ( ix >= struct_grid.G_size_x ) return;

    {
        for ( int iy = ind_thread; iy < struct_grid.G_size_y; iy += N_thread )
        {            
            int ind = ix * struct_grid.Ly_pad + iy;

            T [ind]  = (double) Sxx [ind] * P1 [ind] * (double) Sxx [ind]
                     + (double) Syy [ind] * P1 [ind] * (double) Syy [ind]
                     + (double) Sxx [ind] * P2 [ind] * (double) Syy [ind];
        }
    }

} // cuda_secure_interface_x ()



__global__ void weighted_square_SINGLE_grid ( cuda_Struct_Grid struct_grid , 
                                              ns_type::cuda_precision * S , 
                                              double * P , 
                                              double * T )  // T stores the intermediate output 
{    
    // int N_block  =  gridDim.x;    // Not needed because we don't loop through the (simulation) gridLines (one threadblock each gridLine)
    int N_thread = blockDim.x;

    int ind_block  =  blockIdx.x;
    int ind_thread = threadIdx.x;

    int ix = ind_block;
    if ( ix >= struct_grid.G_size_x ) return;

    {
        for ( int iy = ind_thread; iy < struct_grid.G_size_y; iy += N_thread )
        {            
            int ind = ix * struct_grid.Ly_pad + iy;
            T [ind] = (double) S [ind] * P [ind] * (double) S [ind];
        }
    }

} // cuda_secure_interface_x ()
