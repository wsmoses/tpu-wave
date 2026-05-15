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
    Grid.set_grid_parameters ( {'N','N'} , true );
    Grid.set_forward_operators ();
    Grid.define_parameters_energy ();
    Grid.adjust_parameters_energy_periodic ();

    return 0;
}

