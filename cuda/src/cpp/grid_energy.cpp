#include "grid.hpp"
// This function contains grid member functions related to energy calculation.


// NOTE: Assume all boundaries are weak


void Class_Grid::define_parameters_energy ()
{
    // if ( (G_type_x - 'M') + (G_type_y - 'M') == ('N' - 'M') * 2 )  // NN (Sxx;Syy) grid
    if ( strcmp( grid_name.c_str(), "SMM" ) == 0 ) 
    {
        run_time_vector<ns_type::host_precision> & V_LAMBDA    = this->Vec_prmt.at(0);
        run_time_vector<ns_type::host_precision> & V_2_MU      = this->Vec_prmt.at(1);

        run_time_vector< double > & Energy_V_P1 = this->Vec_prmt_enrg.at(0);    // (lambda + 2*mu) / (4*mu * (lambda+mu))
        run_time_vector< double > & Energy_V_P2 = this->Vec_prmt_enrg.at(1);    // -lambda         / (4*mu * (lambda+mu))

        for ( int i = 0; i < V_LAMBDA.length; i++ )
        {
            Energy_V_P1.at(i) =   ( (double) V_LAMBDA.at(i) + (double) V_2_MU.at(i) ) 
                                / ( (double) V_2_MU.at(i) * ( 2. * (double) V_LAMBDA.at(i) + (double) V_2_MU.at(i) ) );
            Energy_V_P2.at(i) = - ( (double) V_LAMBDA.at(i) * 2.                    ) 
                                / ( (double) V_2_MU.at(i) * ( 2. * (double) V_LAMBDA.at(i) + (double) V_2_MU.at(i) ) );
            // NOTE: * 2. for the same reason that 1/mu is used for {Sxy} instead of 1/(2*mu) - the parameter should be used 
            //       twice (symmetrically), but we only do one calculation with it.
        }
    }

    // if ( (G_type_x - 'M') + (G_type_y - 'M') == ('N' - 'M') * 1 )  // MN (Vx) or NM (Vy) grid
    if ( strcmp( grid_name.c_str(), "Vx" ) == 0 || strcmp( grid_name.c_str(), "Vy" ) == 0 ) 
    {
        run_time_vector<ns_type::host_precision> &        V_RHO_VM = this->Vec_prmt     .at(0);
        run_time_vector< double >              & Energy_V_RHO_VM = this->Vec_prmt_enrg.at(0);

        for ( int i = 0; i < V_RHO_VM.length; i++ ) { Energy_V_RHO_VM.at(i) = V_RHO_VM.at(i); }
    }

    // if ( (G_type_x - 'M') + (G_type_y - 'M') == ('N' - 'M') * 0 )  // MM (Sxy) grid
    if ( strcmp( grid_name.c_str(), "Sxy" ) == 0 ) 
    {
        run_time_vector<ns_type::host_precision> &                 V_MU_Sxy = this->Vec_prmt     .at(0);
        run_time_vector< double >              & Energy_V_ONE_over_MU_Sxy = this->Vec_prmt_enrg.at(0);

        for ( int i = 0; i < V_MU_Sxy.length; i++ ) { Energy_V_ONE_over_MU_Sxy.at(i) = 1. / (double) V_MU_Sxy.at(i); }
    }

} // define_parameters_energy ()
//
// NOTE: Decided not to call adjust_parameters_energy () inside define_parameters_energy () as 
//       an effort to reduce the difficulty to track things down and to improve expressiveness.


// NOTE: The following function adjusts the parameters related to energy calculation 
//       on this grid - the norm matrices are multplied to the parameters.
//           Since all boundaries are considered weak, there is NO need to adjust for 
//       duplication on N grids anymore.
void Class_Grid::adjust_parameters_energy_periodic ()
{
    using ns_input::dx;
    
    // NOTE: The suffix "periodic" means periodic boundary condition is imposed on y direction;
    //       x direction is still associated with free surface boundary condition.

    for ( int i_p = 0; i_p < 1; i_p++ )
    {
        run_time_vector< double > & P = Vec_prmt_enrg.at(0);

        for ( const char& c_dir : {'x'} )
        {
            auto & A_diag_L = Map_A_bdry_diag.at( { c_dir , 'L' } );
            auto & A_diag_R = Map_A_bdry_diag.at( { c_dir , 'R' } );

            int ix;
            int iy;

            int & i_dir = c_dir == 'x' ? ix : iy;


            // ---- loop bounds
            const int LFT_bound_BGN_x = 0;  const int LFT_bound_END_x = c_dir == 'x' ? A_diag_L.length : G_size_x;
            const int LFT_bound_BGN_y = 0;  const int LFT_bound_END_y = c_dir == 'y' ? A_diag_L.length : G_size_y;

            // adjust for left bdry points
            for ( ix = LFT_bound_BGN_x; ix < LFT_bound_END_x; ix++ )
            for ( iy = LFT_bound_BGN_y; iy < LFT_bound_END_y; iy++ )
            {
                int i_v = ix * G_size_y + iy;
                P.at(i_v) = 0;
            }


            // ---- loop bounds
            const int INT_bound_BGN_x = c_dir == 'x' ? A_diag_L.length : 0;  const int INT_bound_END_x = c_dir == 'x' ? G_size_x - A_diag_R.length : G_size_x;
            const int INT_bound_BGN_y = c_dir == 'y' ? A_diag_L.length : 0;  const int INT_bound_END_y = c_dir == 'y' ? G_size_y - A_diag_R.length : G_size_y;

            // adjust for interior points
            for ( ix = INT_bound_BGN_x; ix < INT_bound_END_x; ix++ )
            for ( iy = INT_bound_BGN_y; iy < INT_bound_END_y; iy++ )
            {
                int i_v = ix * G_size_y + iy;
                P.at(i_v) = 0;
            }


            // ---- loop bounds
            const int RHT_bound_BGN_x = c_dir == 'x' ? G_size_x - A_diag_R.length : 0;  const int RHT_bound_END_x = G_size_x;
            const int RHT_bound_BGN_y = c_dir == 'y' ? G_size_y - A_diag_R.length : 0;  const int RHT_bound_END_y = G_size_y;

            const int RHT_bound_BGN_dir = Map_G_size.at(c_dir) - A_diag_R.length;

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


double Class_Grid::energy_calculation ()
{
    double E = 0.;

    // if ( (G_type_x - 'M') + (G_type_y - 'M') == ('N' - 'M') * 2 )  // NN grid
    if ( strcmp( grid_name.c_str(), "SMM" ) == 0 ) 
    {
        // NOTE: the following S is from THIS grid
        run_time_vector<ns_type::host_precision> & Sxx = Vec_soln.at(0);
        run_time_vector<ns_type::host_precision> & Syy = Vec_soln.at(1);

        run_time_vector< double > & P1 = Vec_prmt_enrg.at(0);
        run_time_vector< double > & P2 = Vec_prmt_enrg.at(1);

        for ( int i = 0; i < G_size_x * G_size_y; i++ ) 
        { 
            E = E + (double) Sxx(i) * P1(i) * (double) Sxx(i);
            E = E + (double) Syy(i) * P1(i) * (double) Syy(i);

            E = E + (double) Sxx(i) * P2(i) * (double) Syy(i);
        }
    }
    else
    {
        // NOTE: the following S is from THIS grid
        run_time_vector<ns_type::host_precision> & S = Vec_soln.at(0);

        run_time_vector< double > & P = Vec_prmt_enrg.at(0);

        for ( int i = 0; i < S.length; i++ ) 
            { E = E + (double) S(i) * P(i) * (double) S(i); }
    }

    // printf("%c %c : %16.15e \n" , G_type_x, G_type_y, E/2.);

    return E/2.;
}
