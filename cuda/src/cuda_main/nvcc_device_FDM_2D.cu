#include "input_processing.hpp"

#include "grid.cuh"
#include "namespace_device_variable.cuh"  /* probably dummy, only the #include "namespace_type.cuh" inside it is meaningful */

int main(int argc, char* argv[]) 
{
    using ns_forward::N_dir;
    using namespace ns_input;

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
	    bool_energy = true;


    ns_input_derived_variables();
    checking_and_printouts_input_parameters();


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
