
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
    ns_type::cuda_precision * Sxx_MM = nullptr;
    ns_type::cuda_precision * Syy_MM = nullptr;
    double * P1_MM = nullptr;
    double * P2_MM = nullptr;
    double * T_MM = nullptr;

    cudaMalloc( &Sxx_MM, 369664 * sizeof(ns_type::cuda_precision) );
    cudaMalloc( &Syy_MM, 369664 * sizeof(ns_type::cuda_precision) );
    cudaMalloc( &P1_MM, 369664 * sizeof(double) );
    cudaMalloc( &P2_MM, 369664 * sizeof(double) );
    cudaMalloc( &T_MM, 369664 * sizeof(double) );

    // Grid SNN variables
    ns_type::cuda_precision * S_NN = nullptr;
    double * P_NN = nullptr;
    double * T_NN = nullptr;

    cudaMalloc( &S_NN, 369664 * sizeof(ns_type::cuda_precision) );
    cudaMalloc( &P_NN, 369664 * sizeof(double) );
    cudaMalloc( &T_NN, 369664 * sizeof(double) );

    for (int it=0; it<Nt; it++) 
    {
        // Inlined energy_calculation for grid_SMM
        {
            weighted_square_NORMAL_grid <<< 600 , 32 >>> ( Sxx_MM , Syy_MM , P1_MM , P2_MM , T_MM, 600, 600, 608 );

            double * d_result = nullptr;
            cudaMalloc( &d_result , sizeof(double) );
            single_thread_reduce <<< 1 , 1 >>> ( T_MM , d_result, 369664 );
            cudaFree( d_result );
        }

        // Inlined energy_calculation for grid_SNN
        {
            weighted_square_SINGLE_grid <<< 601 , 32 >>> ( S_NN , P_NN , T_NN, 601, 601, 608 );

            double * d_result = nullptr;
            cudaMalloc( &d_result , sizeof(double) );
            single_thread_reduce <<< 1 , 1 >>> ( T_NN , d_result, 369664 );
            cudaFree( d_result );
        }

    }  // for (int it=0; it<Nt; it++) 

    cudaFree( Sxx_MM );
    cudaFree( Syy_MM );
    cudaFree( P1_MM );
    cudaFree( P2_MM );
    cudaFree( T_MM );
    cudaFree( S_NN );
    cudaFree( P_NN );
    cudaFree( T_NN );
    return 0;
}
