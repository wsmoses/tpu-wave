#include "forward.hpp"
#include "inverse.hpp"

#include "input_processing.hpp"

#include <filesystem>

// #include "mpi.h"


int main(int argc, char* argv[]) 
{
    using ns_forward::N_dir;
    using namespace ns_input;

    Class_Forward_Specs Fwd_Specs;
    Class_Inverse_Specs Inv_Specs;

    // processing the command line option to get the medium_name 
    // (which determines the input file and data to read in)
    command_line_input_processing( argc, argv, "medium" );
    std::string file_name = "../../data/" + medium_name + "/InputFile.txt";

    file_input_processing( file_name );

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
    //       
    // [2024/04/03]
    // NOTE: We should really try to rid of "XY_to_01 ()" - based on the "principle":
    //       maintainable code over performant code.

    // --------------------------------------------------- //
    // --------------- Numerical parameters -------------- //
    // --------------------------------------------------- //

    dx = static_cast<double>( num_dx_soln ) / static_cast<double>( den_dx_soln );
    dt = dt_max * CFL_constant; 
    // we need more comments above on the types


    // {   
    //     // double max_velocity = 2.;
    //     // Fwd_Specs.dt = 0.625 * ( ( dx / max_velocity ) / sqrt( static_cast<double>( N_dir ) ) );
    //     // Fwd_Specs.dt = 1e-4;
    // }


    // -------------------------------------------------- //
    // --------------- Define the classes --------------- //
    // -------------------------------------------------- //

    if ( bool_energy ) { ns_input::Record_E_k .allocate_memory (Nt); }
    if ( bool_energy ) { ns_input::Record_E_p0.allocate_memory (Nt); }
    if ( bool_energy ) { ns_input::Record_E_p1.allocate_memory (Nt); }

    std::array< Class_Grid , 2<<(N_dir-1) > Grids;    // NOTE: use bit shift to take the power of 2 (only works when base is 2); 
                                                      //       reason - std::pow () promotes int to float.

    constexpr auto Array_Grid_types = ns_forward::define_Array_Grid_types ();
    if ( Array_Grid_types.size() != Grids.size() ) 
    { 
        printf( "Array_Grid_types size and Grids size do not agree %ld %ld.\n", Array_Grid_types.size(), Grids.size() ); 
        fflush(stdout); exit(0); 
    }

    // ---- Assign to the map from grid types to pointers of Class_Grid
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(0)] = &Grids.at(0);
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(1)] = &Grids.at(1);
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(2)] = &Grids.at(2);
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(3)] = &Grids.at(3);

    if ( Fwd_Specs.Map_Grid_pointers.size() != Grids.size() ) 
        { printf( "Map_Grid_pointers is supposed to have size %d.\n", 2<<(N_dir-1) ); fflush(stdout); exit(0); }


    for ( const auto & iter_grid_type : Array_Grid_types ) 
        { Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->set_grid_parameters ( iter_grid_type , bool_energy ); }

    for ( const auto & iter_grid_type : Array_Grid_types ) 
        { Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->set_forward_operators (); }

    for ( const auto & iter_grid_type : Array_Grid_types ) 
        { Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->set_grid_pointers ( Fwd_Specs.Map_Grid_pointers ); }


    // -------------------------------------------------------------- //
    // ---------------- Input the physical parameters --------------- //
    // -------------------------------------------------------------- //

    // NOTE: Just place the input files in a different folder can lead to 10%-20% 
    //       performance difference;
    // NOTE: Just add printfS (outside of the critical loop) can lead to 10^-12 
    //       level discrepancies in the printouts.

/*
    // Inverse parameters are stored in Inv_Specs.
    for ( const std::string prmt_name : { "rho" , "vp" , "vs" } )
    {
        file_name = "../../misc/generate_parameters/output/" + medium_name + "/" + prmt_name + ".bin";
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

    // ---- verification (overwrite the above readin parameter data)
    for ( const std::string prmt_name : { "rho" , "vp" , "vs" } )
        { Inv_Specs.Map_inv_prmt[ prmt_name ].allocate_memory ( ns_input::PADDED_inv_prmt_size ); }

    Inv_Specs.Map_inv_prmt.at("rho").set_constant(1);
    Inv_Specs.Map_inv_prmt.at("vp" ).set_constant(2);
    Inv_Specs.Map_inv_prmt.at("vs" ).set_constant(1);
    // ---- verification
    // NOTE: I think the three parameters are indeed constant 1, 2, and 1 in the stored file.


    // Forward parameters are stored in Grids (interpolated from the inverse parameters).
    for ( const auto & iter_grid_type : Array_Grid_types ) 
        { Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->retrieve_forward_parameter ( Inv_Specs ); }

    // NOTE: max value of a parameter P can be retrieved using * std::max_element( P.begin() , P.end() );
    //       pay particular ATTENTION to the dereferencing operator *.

    if ( bool_energy )
    {
        for ( const auto & iter_grid_type : Array_Grid_types ) 
            { Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->define_parameters_energy (); }
        // for ( auto & iter_grid : Grids ) { iter_grid.define_parameters_energy (); }

        // DO ONE THING WELL
        for ( const auto & iter_grid_type : Array_Grid_types ) 
            { Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->adjust_parameters_energy_periodic (); }
    }

    for ( const auto & iter_grid_type : Array_Grid_types )
    {
        auto & grid = Fwd_Specs.Map_Grid_pointers.at(iter_grid_type);
        if ( strcmp( grid->grid_name.c_str(), "Vx" ) == 0 || strcmp( grid->grid_name.c_str(), "Vy" ) == 0 )
            { grid->make_density_reciprocal ( Inv_Specs ); }
    }


    print_discretization_parameters ( 1. , 2. );


    // ---------------------------------------------------------------------------- //
    // ---------------- Process the source and receiver information --------------- //
    // ---------------------------------------------------------------------------- //

    std::vector< struct_src_input >                   & Vec_Src_Input     = Inv_Specs.Vec_Src_Input;
    std::map< int , std::vector< struct_rcv_input > > & Map_Vec_Rcv_Input = Inv_Specs.Map_Vec_Rcv_Input;
    // NOTE: In the above type definition for Map_Vec_Rcv_Input, the map key index through src_index; 
    //       Vec index through rcv_index for this src_index. 
    //           Decided to use Map instead of Vec since src_index may not be numbered consecutively, 
    //       or not start with zero.

    if ( !Vec_Src_Input.empty() ) 
        { printf("Vec_Src_Input should be empty to start with.\n"); fflush(stdout); exit(0); }
    if ( !Map_Vec_Rcv_Input.empty() ) 
        { printf("Map_Vec_Rcv_Input should be empty to start with.\n"); fflush(stdout); exit(0); }
    
    // The input file from where to read in the source and receiver information
    file_name = "../../data/" + medium_name + "/SrcRcv.txt";
    Inv_Specs.input_SrcRcv_locations( file_name );

    Inv_Specs.data_misfit = 0.;  // (re)set the aggregated data misfit for all sources to zero before 
                                 // enter the loop that goes through the sources and lauches simulations 

    for ( auto & iter_vec : Vec_Src_Input )  // NOTE: this for loop will disappear (naturally) in the MPI environment
    {
        // ---- Process the source and receiver location 

        printf( "Processing source locations for source %d.\n", iter_vec.src_index );
        Fwd_Specs.process_src_locations ( iter_vec );           // NOTE: This function should return some information so that 
                                                                //       I can know if the src process is successful.
        int & fwd_src_index = Fwd_Specs.src_forward.src_index;


        printf( "    Processing receiver locations for source %d.\n", fwd_src_index );
        if ( Map_Vec_Rcv_Input.at( fwd_src_index ).size() <= 0 )
            { printf( "No receiver found for source %d.\n", fwd_src_index ); fflush(stdout); exit(0); }

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


        // ---- reset solution fields to zero
        for ( const auto & iter_grid_type : Array_Grid_types ) 
        { 
            for ( int i_field = 0; i_field < Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->N_soln; i_field++ )
                { Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->Vec_soln.at(i_field).set_constant(0.); }
        }
    }  

    return 0;
}
