#include <vector>
#include <string>
#include "namespace_input.hpp"
#include "namespace_device_variable.cuh"  /* probably dummy, only the #include "namespace_type.cuh" inside it is meaningful */

int main(int argc, char* argv[]) 
{

	ns_input::InputParams params;
    params.Mx_prmt = 600;
    params.My_prmt = 600;
    params.Mx_soln = 600;
    params.My_soln = 600;
    params.Nt = 60000;
    params.dt_max = 1.6e-04;
    params.CFL_constant = 0.625;
    params.central_f = 5.0;
    params.time_delay = 0.25;
    params.dx = 0.008;
    params.dt = 1.6e-04;

    params.Nx_prmt = 601;
    params.Ny_prmt = 601;
    params.Nx_soln = 601;
    params.Ny_soln = 601;
    
    params.inv_prmt_stride_x = 601;
    params.inv_prmt_stride_y = 1;
    params.inv_prmt_size = 361201;
    params.PADDED_inv_prmt_size = 362404;

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
    run_time_vector<ns_type::host_precision> Vec_prmt { 601 * 601 };
    run_time_vector< double > Vec_prmt_enrg { 601 * 601 };

    run_time_vector<ns_type::host_precision> stencil_dt_dx {4};
    for ( int i=0; i<4; i++ ) 
        { stencil_dt_dx.at(i) = ns_forward::stencil.at(i) * 0.02; }
        
    int stride_x = 601;
    int stride_y = 1;

    // Inlined define_parameters_energy()
    {
        run_time_vector<ns_type::host_precision> & V_MU_Sxy = Vec_prmt;
        run_time_vector< double > & Energy_V_ONE_over_MU_Sxy = Vec_prmt_enrg;
        for ( int i = 0; i < V_MU_Sxy.length; i++ ) { Energy_V_ONE_over_MU_Sxy.at(i) = 1. / (double) V_MU_Sxy.at(i); }
    }

    // Inlined adjust_parameters_energy_periodic()
    {
        double dx = params.dx;
        
        for ( int i_p = 0; i_p < 1; i_p++ )
        {
            run_time_vector< double > & P = Vec_prmt_enrg;

            for ( const char& c_dir : {'x'} )
            {
                int A_diag_L_length = ns_forward::A_diag.N_L.length;
                int A_diag_R_length = ns_forward::A_diag.N_R.length;

                int ix;
                int iy;

                int & i_dir = ix;


                // ---- loop bounds
                const int LFT_bound_BGN_x = 0;  const int LFT_bound_END_x = A_diag_L_length;
                const int LFT_bound_BGN_y = 0;  const int LFT_bound_END_y = G_size_y;

                // adjust for left bdry points
                for ( ix = LFT_bound_BGN_x; ix < LFT_bound_END_x; ix++ )
                for ( iy = LFT_bound_BGN_y; iy < LFT_bound_END_y; iy++ )
                {
                    int i_v = ix * G_size_y + iy;
                    P.at(i_v) = 0;
                }


                // ---- loop bounds
                const int INT_bound_BGN_x = A_diag_L_length;  const int INT_bound_END_x = G_size_x - A_diag_R_length;
                const int INT_bound_BGN_y = 0;  const int INT_bound_END_y = G_size_y;

                // adjust for interior points
                for ( ix = INT_bound_BGN_x; ix < INT_bound_END_x; ix++ )
                for ( iy = INT_bound_BGN_y; iy < INT_bound_END_y; iy++ )
                {
                    int i_v = ix * G_size_y + iy;
                    P.at(i_v) = 0;
                }


                // ---- loop bounds
                const int RHT_bound_BGN_x = G_size_x - A_diag_R_length;  const int RHT_bound_END_x = G_size_x;
                const int RHT_bound_BGN_y = 0;  const int RHT_bound_END_y = G_size_y;

                const int RHT_bound_BGN_dir = G_size_x - A_diag_R_length;

                // adjust for right bdry points
                for ( ix = RHT_bound_BGN_x; ix < RHT_bound_END_x; ix++ )
                for ( iy = RHT_bound_BGN_y; iy < RHT_bound_END_y; iy++ )
                {
                    int i_v = ix * G_size_y + iy;
                    P.at(i_v) = 0;
                }
            }

            
            // ---- y direction
            {
                // adjust for interior points
                for ( int ix = 0; ix < G_size_x; ix++ )
                {
                    for ( int iy = 0; iy < G_size_y; iy++ )
                    {
                        int i_v = ix * G_size_y + iy;
                        P.at(i_v) = P.at(i_v) * dx;
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
