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

    Class_Forward_Specs Fwd_Specs;
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


    // {   
    //     // double max_velocity = 2.;
    //     // Fwd_Specs.dt = 0.625 * ( ( dx / max_velocity ) / sqrt( static_cast<double>( N_dir ) ) );
    //     // Fwd_Specs.dt = 1e-4;
    // }


    // -------------------------------------------------- //
    // --------------- Define the classes --------------- //
    // -------------------------------------------------- //

    std::array< Class_Grid , 2<<(N_dir-1) > Grids;    // NOTE: use bit shift to take the power of 2 (only works when base is 2); 
                                                      //       reason - std::pow () promotes int to float.

    constexpr auto Array_Grid_types = ns_forward::define_Array_Grid_types ();
    if ( Array_Grid_types.size() != Grids.size() )  // change to assert
    { 
        printf( "Array_Grid_types size and Grids size do not agree %ld %ld.\n", Array_Grid_types.size(), Grids.size() ); 
        fflush(stdout); exit(0); 
    }

    // ---- Assign to the map from grid types to pointers of Class_Grid
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(0)] = &Grids.at(0);
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(1)] = &Grids.at(1);
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(2)] = &Grids.at(2);
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(3)] = &Grids.at(3);

    if ( Fwd_Specs.Map_Grid_pointers.size() != Grids.size() )  // change to assert
        { printf( "Map_Grid_pointers is supposed to have size %d.\n", 2<<(N_dir-1) ); fflush(stdout); exit(0); }


    for ( const auto & iter_grid_type : Array_Grid_types ) 
        { Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->set_grid_parameters ( iter_grid_type , bool_energy ); }

    for ( const auto & iter_grid_type : Array_Grid_types ) 
        { Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->set_forward_operators (); }

    for ( const auto & iter_grid_type : Array_Grid_types ) 
        { Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->set_grid_pointers ( Fwd_Specs.Map_Grid_pointers ); }


    cuda_Class_Grid<'N', 'N', 601, 601> grid_SNN;
    cuda_Class_Grid<'M', 'M', 600, 600> grid_SMM;

		grid_SMM.cuda_Class_Grid_initialize ( &Grids.at(3) );
		grid_SNN.cuda_Class_Grid_initialize ( &Grids.at(0) );
    // NOTE: copy the parameters from hst to dev to start the simulation on device
    for ( cuda_Class_Grid_Base * cuda_class_grid : {(cuda_Class_Grid_Base *)&grid_SMM, (cuda_Class_Grid_Base *)&grid_SNN} ) 
    {
        for ( int i_field = 0; i_field < cuda_class_grid->N_prmt; i_field++ )
        { 
            cuda_class_grid->copy_field_pitched ( "prmt" , i_field , "hst_to_dev" ); 
            
            if ( bool_energy )
                { cuda_class_grid->copy_field_pitched_enrg ( "enrg" , i_field , "hst_to_dev" ); }
        }
    }




// ---- function body of forward_simulation is copied below so that 
//      we can experiment incrementally (one kernel at a time)
{

    for (int it=0; it<Nt; it++) 
    {


		grid_SMM.energy_calculation ();
		grid_SNN.energy_calculation ();

    }  // for (int it=0; it<Nt; it++) 


} 

    return 0;
}
