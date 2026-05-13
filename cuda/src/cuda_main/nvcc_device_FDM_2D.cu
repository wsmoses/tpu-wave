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

    cuda_Class_Grid<'N', 'N', 601, 601> grid_SNN;
    cuda_Class_Grid<'M', 'M', 600, 600> grid_SMM;

    grid_SMM.cuda_Class_Grid_initialize ( "SMM", 2, 2 );
    grid_SNN.cuda_Class_Grid_initialize ( "Sxy", 1, 1 );
{

    for (int it=0; it<Nt; it++) 
    {


        // Inlined energy_calculation for grid_SMM
        {
            auto & Sxx = grid_SMM.Vec_soln[0];
            auto & Syy = grid_SMM.Vec_soln[1];
            auto & P1  = grid_SMM.Vec_prmt_enrg[0];
            auto & P2  = grid_SMM.Vec_prmt_enrg[1];
            auto & T   = grid_SMM.thrust_memory;

            weighted_square_NORMAL_grid <cuda_Struct_Grid<'M', 'M', 600, 600>> <<< 600 , 32 >>> ( Sxx.ptr , Syy.ptr , P1.ptr , P2.ptr , T.ptr );

            double * d_result = nullptr;
            cudaMalloc( &d_result , sizeof(double) );
            single_thread_reduce <cuda_Struct_Grid<'M', 'M', 600, 600>::length_memory> <<< 1 , 1 >>> ( T.ptr , d_result );
            cudaFree( d_result );
        }

        // Inlined energy_calculation for grid_SNN
        {
            auto & S = grid_SNN.Vec_soln[0];
            auto & P = grid_SNN.Vec_prmt_enrg[0];
            auto & T = grid_SNN.thrust_memory;

            weighted_square_SINGLE_grid <cuda_Struct_Grid<'N', 'N', 601, 601>> <<< 601 , 32 >>> ( S.ptr , P.ptr , T.ptr );

            double * d_result = nullptr;
            cudaMalloc( &d_result , sizeof(double) );
            single_thread_reduce <cuda_Struct_Grid<'N', 'N', 601, 601>::length_memory> <<< 1 , 1 >>> ( T.ptr , d_result );
            cudaFree( d_result );
        }

    }  // for (int it=0; it<Nt; it++) 


} 

    return 0;
}
