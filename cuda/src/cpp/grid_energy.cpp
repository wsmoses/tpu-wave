#include "grid.hpp"
// This function contains grid member functions related to energy calculation.


// NOTE: Assume all boundaries are weak


 // define_parameters_energy ()//
// NOTE: Decided not to call adjust_parameters_energy () inside define_parameters_energy () as 
//       an effort to reduce the difficulty to track things down and to improve expressiveness.


// NOTE: The following function adjusts the parameters related to energy calculation 
//       on this grid - the norm matrices are multplied to the parameters.
//           Since all boundaries are considered weak, there is NO need to adjust for 
//       duplication on N grids anymore.
void Class_Grid::adjust_parameters_energy_periodic ()
{
    double dx = params.dx;
    
    // NOTE: The suffix "periodic" means periodic boundary condition is imposed on y direction;
    //       x direction is still associated with free surface boundary condition.

    for ( int i_p = 0; i_p < 1; i_p++ )
    {
        run_time_vector< double > & P = Vec_prmt_enrg.at(0);

        for ( const char& c_dir : {'x'} )
        {
            int A_diag_L_length = (G_type_x == 'N' ? ns_forward::A_diag.N_L.length : ns_forward::A_diag.M_L.length);
            int A_diag_R_length = (G_type_x == 'N' ? ns_forward::A_diag.N_R.length : ns_forward::A_diag.M_R.length);

            int ix;
            int iy;

            int & i_dir = c_dir == 'x' ? ix : iy;


            // ---- loop bounds
            const int LFT_bound_BGN_x = 0;  const int LFT_bound_END_x = c_dir == 'x' ? A_diag_L_length : G_size_x;
            const int LFT_bound_BGN_y = 0;  const int LFT_bound_END_y = c_dir == 'y' ? A_diag_L_length : G_size_y;

            // adjust for left bdry points
            for ( ix = LFT_bound_BGN_x; ix < LFT_bound_END_x; ix++ )
            for ( iy = LFT_bound_BGN_y; iy < LFT_bound_END_y; iy++ )
            {
                int i_v = ix * G_size_y + iy;
                P.at(i_v) = 0;
            }


            // ---- loop bounds
            const int INT_bound_BGN_x = c_dir == 'x' ? A_diag_L_length : 0;  const int INT_bound_END_x = c_dir == 'x' ? G_size_x - A_diag_R_length : G_size_x;
            const int INT_bound_BGN_y = c_dir == 'y' ? A_diag_L_length : 0;  const int INT_bound_END_y = c_dir == 'y' ? G_size_y - A_diag_R_length : G_size_y;

            // adjust for interior points
            for ( ix = INT_bound_BGN_x; ix < INT_bound_END_x; ix++ )
            for ( iy = INT_bound_BGN_y; iy < INT_bound_END_y; iy++ )
            {
                int i_v = ix * G_size_y + iy;
                P.at(i_v) = 0;
            }


            // ---- loop bounds
            const int RHT_bound_BGN_x = c_dir == 'x' ? G_size_x - A_diag_R_length : 0;  const int RHT_bound_END_x = G_size_x;
            const int RHT_bound_BGN_y = c_dir == 'y' ? G_size_y - A_diag_R_length : 0;  const int RHT_bound_END_y = G_size_y;

            const int RHT_bound_BGN_dir = (c_dir == 'x' ? G_size_x : G_size_y) - A_diag_R_length;

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

            if ( G_type_y == 'N' )
            {
                int iy = 0;
                for ( int ix = 0; ix < G_size_x; ix++ )
                {
                    int i_v = ix * G_size_y + iy;
                    P.at(i_v) = P.at(i_v) * 1.; // 1/2; // Oooo, please, tripped by integer division again ? 03/23/2022
                }
            }

        }

        // Warning: Because of the strongly imposed periodic boundary condition, the first and last 
        //          grid points on y direction are supposed to be duplicates of each other. Therefore, 
        //          when applying the source, if the source happens to land on the periodic boundary, 
        //          it needs to be applied on both ends as below.
        // 
        //              However, because of the special index mapping I am using, the last grid point
        //          was mapped to the first one when being used to calculate derivatives. Therefore, 
        //          although the results on the last grid point is incorrect, the simulation results
        //          elsewhere are not affected since its value is not used.
        //
        //              When calculating energy, if we adjust the energy parameters corresponding to 
        //          the first and last grid points using factors 1./2. and 1./2., the result would be 
        //          incorrect; same for factors 0. and 1.; but if using 1. and 0. instead, the result 
        //          would be correct.
        //              
        //              We should "forbid" the source to be placed on the last grid and use factors 1. 
        //          and 0. when calculating energy. This way, we don't need to test if the source is on 
        //          the strong boundary.

        // NOTE: For how the modulo operator maps the indices, see the misc folder c_modulo_operator.

    }
}




