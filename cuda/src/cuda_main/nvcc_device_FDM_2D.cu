#include "input_processing.hpp"

#include "grid.cuh"
#include "namespace_device_variable.cuh"  /* probably dummy, only the #include "namespace_type.cuh" inside it is meaningful */

int main(int argc, char* argv[]) 
{
    using ns_forward::N_dir;
    using namespace ns_input;

    prmt_M_sizes[0] = 600;
    prmt_M_sizes[1] = 600;
    soln_M_sizes[0] = 600;
    soln_M_sizes[1] = 600;
    num_dx_prmt = 8;
    den_dx_prmt = 1000;
    num_dx_soln = 8;
    den_dx_soln = 1000;
    Nt = 60000;
    CFL_constant = 0.625;
    c_energy = 'Y';
	    bool_energy = true;


    ns_input_derived_variables();


    dx = static_cast<double>( num_dx_soln ) / static_cast<double>( den_dx_soln );
    dt = dt_max * CFL_constant; 


    std::array< Class_Grid , 2<<(N_dir-1) > Grids;    // NOTE: use bit shift to take the power of 2 (only works when base is 2); 
                                                      //       reason - std::pow () promotes int to float.
    Class_Grid GridNN;
    Class_Grid GridMM;
    GridNN.set_grid_parameters ( {'N', 'N'} , bool_energy );
    GridMM.set_grid_parameters ( {'M', 'M'} , bool_energy );
    
    GridNN.set_forward_operators (); 
    GridMM.set_forward_operators (); 

    cuda_Class_Grid<'N', 'N', 601, 601> grid_SNN;
    cuda_Class_Grid<'M', 'M', 600, 600> grid_SMM;

		grid_SMM.cuda_Class_Grid_initialize ( &GridMM );
		grid_SNN.cuda_Class_Grid_initialize ( &GridNN );
{

    for (int it=0; it<Nt; it++) 
    {


		grid_SMM.energy_calculation ();
		grid_SNN.energy_calculation ();

    }  // for (int it=0; it<Nt; it++) 


} 

    return 0;
}
