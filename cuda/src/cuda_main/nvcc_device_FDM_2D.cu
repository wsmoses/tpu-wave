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
    
    ns_input_derived_variables( params );

    Class_Grid Grid(params);
    constexpr auto Array_Grid_types = ns_forward::define_Array_Grid_types ();
    Grid.set_grid_parameters ( Array_Grid_types[0] , true );
    Grid.set_forward_operators ();
    Grid.define_parameters_energy ();
    Grid.adjust_parameters_energy_periodic ();

    return 0;
}

