#include "namespace_input.hpp"
#include "namespace_device_variable.cuh"  /* probably dummy, only the #include "namespace_type.cuh" inside it is meaningful */

int main(int argc, char* argv[]) 
{


    // Local variables instead of Class_Grid
    char G_type_x = 'N';
    char G_type_y = 'N';
    bool free_surface_update = false;
    int G_size_x = 601;
    int G_size_y = 601;
    int G_ix_bgn = 0; int G_ix_end = 601;
    int G_iy_bgn = 0; int G_iy_end = 601;
    int N_modulo_x = 600;
    int N_modulo_y = 600;
    int N_soln = 1;
    int N_prmt = 1;
    run_time_vector< double > Vec_prmt_enrg { 601 * 601 };

    int stride_x = 601;
    int stride_y = 1;

    // Inlined define_parameters_energy()
    {
        run_time_vector< double > & Energy_V_ONE_over_MU_Sxy = Vec_prmt_enrg;
        for ( int i = 0; i < Energy_V_ONE_over_MU_Sxy.length; i++ ) { Energy_V_ONE_over_MU_Sxy.at(i) = 1.; }
    }

    // Inlined adjust_parameters_energy_periodic()
    {
        
        for ( int i_p = 0; i_p < 1; i_p++ )
        {
            run_time_vector< double > & P = Vec_prmt_enrg;

            
            // ---- y direction
            {
                // adjust for interior points
                for ( int ix = 0; ix < G_size_x; ix++ )
                {
                    for ( int iy = 0; iy < G_size_y; iy++ )
                    {
                        int i_v = ix * G_size_y + iy;
                        P.at(i_v) = P.at(i_v) * 0.1;
                    }
                }

                int iy = 0;
                for ( int ix = 0; ix < G_size_x; ix++ )
                {
                    int i_v = ix * G_size_y + iy;
                    P.at(i_v) = P.at(i_v) * 1.; // 1/2; // Oooo, please, tripped by integer division again ? 03/23/2022
                }

            }
        }
    }

    return 0;
}
