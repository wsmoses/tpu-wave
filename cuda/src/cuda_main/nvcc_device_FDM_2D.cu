#include "grid.hpp"
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

    Class_Grid Grid(params);
    
    // Inlined set_grid_parameters({'N','N'}, true)
    Grid.G_type_x = 'N';
    Grid.G_type_y = 'N';
    Grid.grid_name = "Sxy";
    Grid.free_surface_update = false;
    Grid.G_size_x = 601;
    Grid.G_size_y = 601;
    Grid.G_ix_bgn = 0; Grid.G_ix_end = 601;
    Grid.G_iy_bgn = 0; Grid.G_iy_end = 601;
    Grid.N_modulo_x = 600;
    Grid.N_modulo_y = 600;
    Grid.N_soln = 1;
    Grid.N_prmt = 1;
    Grid.N_enrg = 1;
    
    Grid.Vec_prmt.push_back( run_time_vector<ns_type::host_precision> { 601 * 601 } );
    Grid.Vec_prmt_enrg.push_back( run_time_vector< double > { 601 * 601 } );

    // Inlined set_forward_operators()
    for ( int i=0; i<4; i++ ) 
        { Grid.stencil_dt_dx.at(i) = ns_forward::stencil.at(i) * 0.02; }
        
    Grid.stride_x = 601;
    Grid.stride_y = 1;

    // Inlined define_parameters_energy()
    {
        run_time_vector<ns_type::host_precision> & V_MU_Sxy = Grid.Vec_prmt.at(0);
        run_time_vector< double > & Energy_V_ONE_over_MU_Sxy = Grid.Vec_prmt_enrg.at(0);
        for ( int i = 0; i < V_MU_Sxy.length; i++ ) { Energy_V_ONE_over_MU_Sxy.at(i) = 1. / (double) V_MU_Sxy.at(i); }
    }

    Grid.adjust_parameters_energy_periodic ();

    return 0;
}
