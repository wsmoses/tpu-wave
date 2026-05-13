
#include <iostream>
#include <vector>
#include <string>
#include <array>

#include "namespace_type.cuh"
#include "cuda_container_run_time.cuh"

__global__ void weighted_square_NORMAL_grid ( ns_type::cuda_precision * Sxx , ns_type::cuda_precision * Syy , 
                                              double * P1  , double * P2  , 
                                              double * T ,
                                              int size_x, int size_y, int Ly_pad )
{    
    int ind_block  =  blockIdx.x;
    int ind_thread = threadIdx.x;

    int ix = ind_block;
    if ( ix >= size_x ) return;

    for ( int iy = 0; iy < size_y; iy += blockDim.x )
    {            
        int actual_iy = iy + ind_thread;
        if (actual_iy < size_y) {
            int ind = ix * Ly_pad + actual_iy;

            T [ind]  = (double) Sxx [ind] * P1 [ind]
                     + (double) Sxx [ind] * P2 [ind] * (double) Syy [ind];
        }
    }
}

__global__ void weighted_square_SINGLE_grid ( ns_type::cuda_precision * S , 
                                              double * P , 
                                              double * T ,
                                              int size_x, int size_y, int Ly_pad )
{    
    int ind_block  =  blockIdx.x;
    int ind_thread = threadIdx.x;

    int ix = ind_block;
    if ( ix >= size_x ) return;

    for ( int iy = 0; iy < size_y; iy += blockDim.x )
    {            
        int actual_iy = iy + ind_thread;
        if (actual_iy < size_y) {
            int ind = ix * Ly_pad + actual_iy;
            T [ind] = S [ind] * P [ind];
        }
    }
}

__global__ void single_thread_reduce ( double * T , double * result, int length )
{
    double sum = 0;
    for ( int i = 0; i < length; i++ ) {
        sum += T[i];
    }
    *result = sum;
}

int main(int argc, char* argv[]) 
{

    int prmt_M_sizes[2] = {600, 600};
    int soln_M_sizes[2] = {600, 600};
    int num_dx_prmt = 8;
    int den_dx_prmt = 1000;
    int num_dx_soln = 8;
    int den_dx_soln = 1000;
    int Nt = 60000;
    double CFL_constant = 0.625;
    char c_energy = 'Y';
    bool bool_energy = true;

    double dx = static_cast<double>( num_dx_soln ) / static_cast<double>( den_dx_soln );
    double dt = 0.0001 * CFL_constant; // Assuming dt_max was 0.0001 based on previous logs

    // Grid SMM variables
    cuda_run_time_vector<ns_type::cuda_precision> Sxx_MM;
    cuda_run_time_vector<ns_type::cuda_precision> Syy_MM;
    cuda_run_time_vector<double> P1_MM;
    cuda_run_time_vector<double> P2_MM;
    cuda_run_time_vector<double> T_MM;

    Sxx_MM.allocate_memory( 369664 );
    Syy_MM.allocate_memory( 369664 );
    P1_MM.allocate_memory( 369664 );
    P2_MM.allocate_memory( 369664 );
    T_MM.allocate_memory( 369664 );

    // Grid SNN variables
    cuda_run_time_vector<ns_type::cuda_precision> S_NN;
    cuda_run_time_vector<double> P_NN;
    cuda_run_time_vector<double> T_NN;

    S_NN.allocate_memory( 369664 );
    P_NN.allocate_memory( 369664 );
    T_NN.allocate_memory( 369664 );

    for (int it=0; it<Nt; it++) 
    {
        // Inlined energy_calculation for grid_SMM
        {
            weighted_square_NORMAL_grid <<< 600 , 32 >>> ( Sxx_MM.ptr , Syy_MM.ptr , P1_MM.ptr , P2_MM.ptr , T_MM.ptr, 600, 600, 608 );

            double * d_result = nullptr;
            cudaMalloc( &d_result , sizeof(double) );
            single_thread_reduce <<< 1 , 1 >>> ( T_MM.ptr , d_result, 369664 );
            cudaFree( d_result );
        }

        // Inlined energy_calculation for grid_SNN
        {
            weighted_square_SINGLE_grid <<< 601 , 32 >>> ( S_NN.ptr , P_NN.ptr , T_NN.ptr, 601, 601, 608 );

            double * d_result = nullptr;
            cudaMalloc( &d_result , sizeof(double) );
            single_thread_reduce <<< 1 , 1 >>> ( T_NN.ptr , d_result, 369664 );
            cudaFree( d_result );
        }

    }  // for (int it=0; it<Nt; it++) 


    return 0;
}
