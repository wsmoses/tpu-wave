#include "grid.hpp"

#include "inverse.hpp"
#include "namespace_input.hpp"

template<typename T>
void Class_Grid::interpolate_forward_parameter ( run_time_vector< double > & inverse_P , 
                                                 run_time_vector< T > & forward_P )
{
    using namespace ns_input;


    for ( int ix_S = 0; ix_S < G_size_x; ix_S++ ) 
    { 
        for ( int iy_S = 0; iy_S < G_size_y; iy_S++ ) 
        { 
            // step 1): determine location in 'physical' units
            long num_x_loc = 1 << 30;  long den_x_loc = -1;
            long num_y_loc = 1 << 30;  long den_y_loc = -1;

            if ( G_type_x == 'N' ) { num_x_loc =             ix_S * num_dx_soln;  den_x_loc =     den_dx_soln; }
            if ( G_type_y == 'N' ) { num_y_loc =             iy_S * num_dx_soln;  den_y_loc =     den_dx_soln; }

            if ( G_type_x == 'M' ) { num_x_loc = ( 2 * ix_S + 1 ) * num_dx_soln;  den_x_loc = 2 * den_dx_soln; }
            if ( G_type_y == 'M' ) { num_y_loc = ( 2 * iy_S + 1 ) * num_dx_soln;  den_y_loc = 2 * den_dx_soln; }


            // step 2): determine ix_M, iy_M (did we assume dxM >= dxS in the following ?)
            long ix_M = ( num_x_loc * den_dx_prmt ) / ( den_x_loc * num_dx_prmt );
            long iy_M = ( num_y_loc * den_dx_prmt ) / ( den_y_loc * num_dx_prmt );

            if ( ix_M >= Nx_prmt ) ix_M = Nx_prmt - 1;
            if ( iy_M >= Ny_prmt ) iy_M = Ny_prmt - 1;
            if ( ix_M < 0 ) ix_M = 0;
            if ( iy_M < 0 ) iy_M = 0;


            // step 3): determine xi, eta
            double xi  = static_cast<double>( num_x_loc * den_dx_prmt ) / static_cast<double>( den_x_loc * num_dx_prmt ) - ix_M;
            double eta = static_cast<double>( num_y_loc * den_dx_prmt ) / static_cast<double>( den_y_loc * num_dx_prmt ) - iy_M;


            // step 4): interpolate the inverse parameter to the (forward) grid
            const int PADDED_stride = Ny_prmt + 1;
            // NOTE: The inverse parameters have been padded with an extra number
            //       so that the following interpolation won't step out of bound.

            double v = (1 - xi) * (1 - eta) * inverse_P( ( ix_M     ) * PADDED_stride + ( iy_M     ) )     // PADDED_stride = Ny_prmt + 1
                     + (1 - xi) * (    eta) * inverse_P( ( ix_M     ) * PADDED_stride + ( iy_M + 1 ) )     // PADDED_stride = Ny_prmt + 1
                     + (    xi) * (1 - eta) * inverse_P( ( ix_M + 1 ) * PADDED_stride + ( iy_M     ) )     // PADDED_stride = Ny_prmt + 1
                     + (    xi) * (    eta) * inverse_P( ( ix_M + 1 ) * PADDED_stride + ( iy_M + 1 ) );    // PADDED_stride = Ny_prmt + 1
                     
            forward_P( ix_S * G_size_y + iy_S ) = v;


            // check if have stepped out of bound
            // if ( !std::isfinite(v) ) { printf("NaN or Inf detected in v.\n"); fflush(stdout); exit(0); }

            if ( ix_M < 0 || ix_M + 1 > Nx_prmt 
              || iy_M < 0 || iy_M + 1 > Ny_prmt ) 
                { printf(" ---- Out-of-bound access. (%d %d)\n", ix_M, iy_M); fflush(stdout); exit(0); }
        }
    }

} // interpolate_forward_parameter


// NOTE: We can fix the types of foward parameters, which are stored in grids. 
//       For example, we can make it always rho, lambda, and mu. 
//           The types of inverse parameters can be made free to change, which
//       are stored in Inv_Specs.
//           The following function would need to be modified if the inverse 
//       parameters are not rho, vp, and vs. For example, they can be rho, 
//       lambda and mu directly, or they can be log of the 'true' parameters.
//           Alternatively, we can convert the actual parameters used in
//       optimization to rho, vp, and vs first, i.e., using rho, vp, and vs 
//       as intermediate variables. The conversion would then be done in 
//       Inv_Specs. This would be confusing if the inverse parameters are rho, 
//       lambda, and mu or their log verions.
void Class_Grid::retrieve_forward_parameter ( Class_Inverse_Specs &Inv_Specs )
{
    if ( Inv_Specs.Map_inv_prmt.count( "rho" ) == 0 ) { printf("Parameter rho not found.\n"); fflush(stdout); exit(0); }
    if ( Inv_Specs.Map_inv_prmt.count( "vp"  ) == 0 ) { printf("Parameter vp  not found.\n"); fflush(stdout); exit(0); }
    if ( Inv_Specs.Map_inv_prmt.count( "vs"  ) == 0 ) { printf("Parameter vs  not found.\n"); fflush(stdout); exit(0); }
    // NOTE: Instead of exit, we can test 
    //           if rho, vp, vs are stored in Inv_Specs; 
    //           if rho, lambda, mu are stored in Inv_Specs; 
    //           if their log versions are stored in Inv_Specs.
    //       Depending on the case, we can branch in the following to use different formulas 
    //       to convert to rho, lambda, and mu that are used in forward simulation.

    run_time_vector< double > & V_RHO = Inv_Specs.Map_inv_prmt.at( std::string("rho") );
    run_time_vector< double > & V_VP  = Inv_Specs.Map_inv_prmt.at( std::string("vp" ) );
    run_time_vector< double > & V_VS  = Inv_Specs.Map_inv_prmt.at( std::string("vs" ) );

    // if ( (G_type_x - 'M') + (G_type_y - 'M') == ('N' - 'M') * 2 )  // {Sxx;Syy}
    if ( strcmp( grid_name.c_str(), "SMM" ) == 0 ) 
    {
        run_time_vector< double > V_LAMBDA ( ns_input::PADDED_inv_prmt_size );
        run_time_vector< double > V_2_MU   ( ns_input::PADDED_inv_prmt_size );

        for ( int i = 0; i < ns_input::PADDED_inv_prmt_size; i++ )
        {
            V_2_MU   (i) = 2 * V_RHO (i) * V_VS (i) * V_VS (i);
            V_LAMBDA (i) =     V_RHO (i) * V_VP (i) * V_VP (i) - V_2_MU (i);
        }

        interpolate_forward_parameter <ns_type::host_precision> ( V_LAMBDA , Vec_prmt.at(0) );
        interpolate_forward_parameter <ns_type::host_precision> ( V_2_MU   , Vec_prmt.at(1) );
    }


    // if ( (G_type_x - 'M') + (G_type_y - 'M') == ('N' - 'M') * 1 )  // {Vx;Vy}
    if ( strcmp( grid_name.c_str(), "Vx" ) == 0 || strcmp( grid_name.c_str(), "Vy" ) == 0 ) 
    {
        interpolate_forward_parameter <ns_type::host_precision> ( V_RHO , Vec_prmt.at(0) );
    }


    // if ( (G_type_x - 'M') + (G_type_y - 'M') == ('N' - 'M') * 0 )  // {Sxy}
    if ( strcmp( grid_name.c_str(), "Sxy" ) == 0 ) 
    {
        run_time_vector< double > V_MU   ( ns_input::PADDED_inv_prmt_size );
        
        for ( int i = 0; i < ns_input::PADDED_inv_prmt_size; i++ )
            { V_MU (i) = V_RHO (i) * V_VS (i) * V_VS (i); }

        interpolate_forward_parameter <ns_type::host_precision> ( V_MU , Vec_prmt.at(0) );
    }

} // retrieve_forward_parameter


// NOTE: Be very careful about when this function is called; it should only be called after "rho"
//       has been used to prepare other parameters such as those used for energy calculation.
void Class_Grid::make_density_reciprocal ( Class_Inverse_Specs &Inv_Specs )
{
    if ( strcmp( this->grid_name.c_str(), "Vx" ) == 0 || strcmp( this->grid_name.c_str(), "Vy" ) == 0 )
    {
        if ( this->N_prmt > 1 ) 
            { printf("Should only have 1 parameter on this grid"); fflush(stdout); exit(0); }

        // // ---- P is potentially already fp16 before taking the reciprocal
        // {
        //     run_time_vector<ns_type::host_precision> & P = this->Vec_prmt.at(0);
        //     for ( int i = 0; i < P.length; i++ ) { P.at(i) = 1. / (double) P.at(i); }
        // }

        // ---- P is at double before taking the reciprocal (if they are fp16 on disk and if there is
        //      no actual interpolation happening, the result should be the same as the one above; we
        //      have just a little bit interpolation on the MN and NM grids.) /* In fp64, using the 
        //      one above and below should give the same outcome, and we indeed observe so; for fp16, 
        //      we observe very little difference for multiple = 1; the printed out "diff (relative)"
        //      is 1.28e-05 above vs 1.25e-05 below; for fp32, we also observe no difference, but 
        //      that may be because we didn't print out enough digits. */) 
        {
            run_time_vector< double > & inverse_P  = Inv_Specs.Map_inv_prmt.at( std::string("rho") );
            run_time_vector< double >   forward_P ( Vec_prmt.at(0).length );
            interpolate_forward_parameter < double > ( inverse_P , forward_P );

            run_time_vector<ns_type::host_precision> & P = this->Vec_prmt.at(0);
            printf("make_density_reciprocal: grid %s, P.length = %d, forward_P.length = %d\n", this->grid_name.c_str(), P.length, forward_P.length);
            fflush(stdout);
            for ( int i = 0; i < P.length; i++ ) { P.at(i) = 1. / forward_P.at(i); }
        }
    }
    else
        { printf("Shouldn't call make_density_reciprocal on this grid.\n"); fflush(stdout); exit(0); }
} 
// [2024/03/29]
// NOTE: It's more or less ok to use P at ns_type::host_precision as the input for 1. / P.at(i)?
//       /* We may need to write 1. / (double) P.at(i). */
//       An alternative is to call interpolate_forward_parameter inside make_density_reciprocal 
//       so that the input P is in double (interpolated from inverse prmt).