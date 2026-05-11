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

    constexpr auto Array_Grid_types = ns_forward::define_Array_Grid_types ();


    // -------------------------------------------------------------- //
    // ---------------- Input the physical parameters --------------- //
    // -------------------------------------------------------------- //

/*
    // Inverse parameters are stored in Inv_Specs.
    for ( const std::string prmt_name : { "rho" , "vp" , "vs" } )
    {
        file_name = "../../data/" + medium_name + "/" + prmt_name + ".bin";
        // NOTE: During testing, trying to keep only one copy of each input file so that
        //       we don't need to worry whether the input files are the intended version.

        Inv_Specs.input_inverse_parameter ( file_name , prmt_name );
    }
    // NOTE: The inverse parameters stored in Inv_Specs have been PADDED with an extra number 
    //       on each direction to avoid the 'ghostly' bug caused by stepping out of bound.
    //           PAY SPECIAL ATTENTION to the size and stride of the inverse parameters 
    //       acutally stored in Inv_Specs : THE SIZE IS 
    //              ns_input::PADDED_inv_prmt_size = ( Nx_model + 1 ) * ( Ny_model + 1 )
    //           THE STRIDE IS 
    //              ( Ny_model + 1 ).
*/
    // for ( const std::string prmt_name : { "rho" , "vp" , "vs" } )
    // {
    //     auto v = Inv_Specs.Map_inv_prmt.at(prmt_name);
    //     printf( "%16.15e %16.15e\n", *std::min_element(v.begin(),v.end())
    //                                , *std::max_element(v.begin(),v.end()) );
    // }
    // [2023/09/24]
    // NOTE: min are zero because of the padding.
    // [2023/09/24]
    // NOTE: min_element and max_element cannot be called on __half.



    // ---- verification (overwrite the above readin parameter data)
    for ( const std::string prmt_name : { "rho" , "vp" , "vs" } )
        { Inv_Specs.Map_inv_prmt[ prmt_name ].allocate_memory ( ns_input::PADDED_inv_prmt_size ); }

    Inv_Specs.Map_inv_prmt.at("rho").set_constant(1);
    Inv_Specs.Map_inv_prmt.at("vp" ).set_constant(2);
    Inv_Specs.Map_inv_prmt.at("vs" ).set_constant(1);
    // ---- verification
    // NOTE: I think the three parameters are indeed constant 1, 2, and 1 in the stored file.


    // [2024/04/03]

    // NOTE: max value of a parameter P can be retrieved using * std::max_element( P.begin() , P.end() );
    //       pay particular ATTENTION to the dereferencing operator *.

    // [2024/04/03]

    // [2024/04/03]
    // NOTE: Let's first see how the results look like before making changes to make_density_reciprocal ().


// -------------------------------- WARNING -------------------------------- //
// [2023/07/18]                                                              //
// NOTE: We assume that density is made reciprocal on cpu at some point for  //
//       both cpu and gpu simulations.                                       //
// -------------------------------- WARNING -------------------------------- //


    // ---------------------------------------------------------------------------- //
    // ---------------- Process the source and receiver information --------------- //
    // ---------------------------------------------------------------------------- //

    std::vector< struct_src_input >                   & Vec_Src_Input     = Inv_Specs.Vec_Src_Input;
    std::map< int , std::vector< struct_rcv_input > > & Map_Vec_Rcv_Input = Inv_Specs.Map_Vec_Rcv_Input;
    // NOTE: In the above type definition for Map_Vec_Rcv_Input, the map key index through src_index; 
    //       Vec index through rcv_index for this src_index. 
    //           Decided to use Map instead of Vec since src_index may not be numbered consecutively, 
    //       or not start with zero.

    Vec_Src_Input.push_back ( struct_src_input { 1 , {'N','N'} , {0,1,0,1} } );
    Map_Vec_Rcv_Input[1] = { struct_rcv_input { 1 , 1 , {'N','N'} , {0,1,0,1} } };

    Inv_Specs.data_misfit = 0.;  // (re)set the aggregated data misfit for all sources to zero before 
                                 // enter the loop that goes through the sources and lauches simulations 

    for ( auto & iter_vec : Vec_Src_Input )  // NOTE: this for loop will disappear (naturally) in the MPI environment
    {
        // ---- Process the source and receiver location 

        Fwd_Specs.process_src_locations ( iter_vec );
        int & fwd_src_index = Fwd_Specs.src_forward.src_index;

        printf( "    Processing receiver locations for source %d.\n", fwd_src_index );
        if ( Map_Vec_Rcv_Input.at( fwd_src_index ).size() <= 0 )
            { printf( "No receiver found for source %d.\n", fwd_src_index ); fflush(stdout); exit(0); }

        printf("Array_Grid_types size: %lu\n", Array_Grid_types.size());
        for (auto const& type : Array_Grid_types) {
            printf("Type: %c %c\n", type[0], type[1]);
        }

        for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_N_rcvs            [grid_type] = 0;  }
        for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_struct_rcv_forward[grid_type] = {}; }
        for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_record_rcv        [grid_type] = {}; }
        for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_RESULT_rcv        [grid_type] = {}; }
        for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_misfit_rcv        [grid_type] = {}; }
        // NOTE: In the above, we initiate an entry for each grid type, regardless of whether 
        //       there is a receiver associated with it. Is this NECESSARY ?
        //           For API stability, decided to keep it this way until strong reasons for 
        //       change emerge.

        Fwd_Specs.process_rcv_locations ( Map_Vec_Rcv_Input.at( fwd_src_index ) );
        // NOTE: comments on the re-initialization involved in the above function 
    }

    return 0;
}
