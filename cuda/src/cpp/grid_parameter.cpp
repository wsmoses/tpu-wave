#include "grid.hpp"

#include "inverse.hpp"
#include "namespace_input.hpp"

template<typename T>
void Class_Grid::interpolate_forward_parameter ( 
                                                 run_time_vector< T > & forward_P )
{
    using namespace ns_input;


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

        interpolate_forward_parameter <ns_type::host_precision> ( Vec_prmt.at(0) );
        interpolate_forward_parameter <ns_type::host_precision> ( Vec_prmt.at(1) );
    }


    // if ( (G_type_x - 'M') + (G_type_y - 'M') == ('N' - 'M') * 1 )  // {Vx;Vy}
    if ( strcmp( grid_name.c_str(), "Vx" ) == 0 || strcmp( grid_name.c_str(), "Vy" ) == 0 ) 
    {
        interpolate_forward_parameter <ns_type::host_precision> ( Vec_prmt.at(0) );
    }


    // if ( (G_type_x - 'M') + (G_type_y - 'M') == ('N' - 'M') * 0 )  // {Sxy}
    if ( strcmp( grid_name.c_str(), "Sxy" ) == 0 ) 
    {
        run_time_vector< double > V_MU   ( ns_input::PADDED_inv_prmt_size );
        
        for ( int i = 0; i < ns_input::PADDED_inv_prmt_size; i++ )
            { V_MU (i) = V_RHO (i) * V_VS (i) * V_VS (i); }

        interpolate_forward_parameter <ns_type::host_precision> ( Vec_prmt.at(0) );
    }

} // retrieve_forward_parameter

