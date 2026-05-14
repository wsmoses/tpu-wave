#include "forward.hpp"
#include "inverse.hpp"

#include "input_processing.hpp"

#include "grid.cuh"
#include "namespace_device_variable.cuh"  /* probably dummy, only the #include "namespace_type.cuh" inside it is meaningful */

#include <filesystem>

int main(int argc, char* argv[]) 
{
    using ns_forward::N_dir;
    using namespace ns_input;

    std::map< std::array<char, ns_forward::N_dir> , Class_Grid * > Fwd_Specs;
    Class_Inverse_Specs Inv_Specs;

#ifndef GRIDFLAG
    std::string input_file_name = "/InputFile.txt";
#endif

#if GRIDFLAG == 1
    std::string input_file_name = "/InputFile_1.txt";
#elif GRIDFLAG == 3
    std::string input_file_name = "/InputFile_3.txt";
#elif GRIDFLAG == 5
    std::string input_file_name = "/InputFile_5.txt";
#endif

    // processing the command line option to get the medium_name 
    // (which determines the input file and data to read in)
    command_line_input_processing( argc, argv, "medium" );
    std::string file_name = "../../data/" + medium_name + input_file_name;  // "/InputFile.txt";    

    file_input_processing( file_name );
	input_file_name = file_name;  // this is so that we can output "input_file_name" at the end 
								  // because file_name will be overwritten later.

    // ---- rescanning the command line options to process the remaining arguments;
    //      rescanning needed such that the arguments specified at command line can 
    //      overwrite those specified in input file;
    command_line_input_processing( argc, argv, "all" );

    
    if ( c_energy == 'N' ) bool_energy = false;
    if ( c_energy == 'Y' ) bool_energy = true;
    assert( bool_energy == true );    

    ns_input_derived_variables();
    checking_and_printouts_input_parameters();

    // [2023/07/12] 
    // NOTE: "file_input_processing ()" and "checking_and_printouts_input_parameters ()" 
    //       are from "input_processing.cpp". Both still use "XY_to_01 ()", which maps 
    //       'X' to 1 and 'Y' to 0. 
    //       
    //       This time around, we should try to get rid of "XY_to_01 ()", particularly
    //       because in this folder we are focusing on the "correctness" of numerics 
    //       (at lower precision).

    // --------------------------------------------------- //
    // --------------- Numerical parameters -------------- //
    // --------------------------------------------------- //

    dx = static_cast<double>( num_dx_soln ) / static_cast<double>( den_dx_soln );
    dt = dt_max * CFL_constant; 

    // dt = 0.002;
    // we need more comments above on the types




    // -------------------------------------------------- //
    // --------------- Define the classes --------------- //
    // -------------------------------------------------- //

    if ( bool_energy ) { ns_input::Record_E_k .allocate_memory (Nt); }
    if ( bool_energy ) { ns_input::Record_E_p0.allocate_memory (Nt); }
    if ( bool_energy ) { ns_input::Record_E_p1.allocate_memory (Nt); }

    std::array< Class_Grid , 2<<(N_dir-1) > Grids;    // NOTE: use bit shift to take the power of 2 (only works when base is 2); 
                                                      //       reason - std::pow () promotes int to float.

    constexpr auto Array_Grid_types = ns_forward::define_Array_Grid_types ();
    Grids[0].set_grid_parameters ( Array_Grid_types[0] , bool_energy );
    Grids[0].set_forward_operators ();
    Grids[0].define_parameters_energy ();
    Grids[0].adjust_parameters_energy_periodic ();

    return 0;
}
