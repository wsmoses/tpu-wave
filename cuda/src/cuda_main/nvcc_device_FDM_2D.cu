#include "input_processing.hpp"

#include "grid.hpp"
#include "namespace_device_variable.cuh"  /* probably dummy, only the #include "namespace_type.cuh" inside it is meaningful */

int main(int argc, char* argv[]) 
{

	std::string input_file_name = "/InputFile.txt";
	std::string file_name = "../../data/homogeneous" + input_file_name;
	file_input_processing( file_name );
	ns_input_derived_variables();

    Class_Grid Grid;
    constexpr auto Array_Grid_types = ns_forward::define_Array_Grid_types ();
    Grid.set_grid_parameters ( Array_Grid_types[0] , true );
    Grid.set_forward_operators ();
    Grid.define_parameters_energy ();
    Grid.adjust_parameters_energy_periodic ();

    return 0;
}
