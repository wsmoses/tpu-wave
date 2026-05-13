#include "grid.cuh"
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
            weighted_square_NORMAL_grid <cuda_Struct_Grid<'M', 'M', 600, 600>> <<< 600 , 32 >>> ( Sxx_MM.ptr , Syy_MM.ptr , P1_MM.ptr , P2_MM.ptr , T_MM.ptr );

            double * d_result = nullptr;
            cudaMalloc( &d_result , sizeof(double) );
            single_thread_reduce <cuda_Struct_Grid<'M', 'M', 600, 600>::length_memory> <<< 1 , 1 >>> ( T_MM.ptr , d_result );
            cudaFree( d_result );
        }

        // Inlined energy_calculation for grid_SNN
        {
            weighted_square_SINGLE_grid <cuda_Struct_Grid<'N', 'N', 601, 601>> <<< 601 , 32 >>> ( S_NN.ptr , P_NN.ptr , T_NN.ptr );

            double * d_result = nullptr;
            cudaMalloc( &d_result , sizeof(double) );
            single_thread_reduce <cuda_Struct_Grid<'N', 'N', 601, 601>::length_memory> <<< 1 , 1 >>> ( T_NN.ptr , d_result );
            cudaFree( d_result );
        }

    }  // for (int it=0; it<Nt; it++) 


    return 0;
}
