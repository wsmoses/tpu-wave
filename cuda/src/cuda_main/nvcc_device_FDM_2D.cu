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

    if ( bool_energy ) { ns_input::Record_E_k .allocate_memory (Nt); }
    if ( bool_energy ) { ns_input::Record_E_p0.allocate_memory (Nt); }
    if ( bool_energy ) { ns_input::Record_E_p1.allocate_memory (Nt); }

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
    // [2024/04/03]
    // NOTE: Let's first see how the results look like before making changes to make_density_reciprocal ().

    
    print_discretization_parameters ( 1. , 2. );


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
    }


    // cudaSetDevice(ns_input::device_number);   // select which device to run; 0 is default;
                           // the number of devices can be retrieved by 
                           // cudaGetDeviceCount(&devicesCount);
// Actually, we set it to -1, which is the reason it causes error on sirius (CUDA 12.9)
// I don't feel device_number should be part of the input file or command line; instead,
// should determine it inside the code based on how many src we have.


    // cudaDeviceSetCacheConfig ( cudaFuncCachePreferL1 );


    cuda_Class_Grid<'N', 'N', 601, 601> grid_SNN;
    cuda_Class_Grid<'M', 'M', 600, 600> grid_SMM;
    cuda_Class_Grid<'M', 'N', 600, 601> grid_SMN;
    cuda_Class_Grid<'N', 'M', 601, 600> grid_SNM;

    std::map< std::array<char, ns_forward::N_dir> , cuda_Class_Grid_Base * > cuda_Map_Class_Grid_pointers  {};    

    // ---- Assign to the map from grid types to pointers of cuda_Class_Grid
    cuda_Map_Class_Grid_pointers [{'N','N'}] = &grid_SNN;
    cuda_Map_Class_Grid_pointers [{'M','M'}] = &grid_SMM;
    cuda_Map_Class_Grid_pointers [{'M','N'}] = &grid_SMN;
    cuda_Map_Class_Grid_pointers [{'N','M'}] = &grid_SNM;


    if ( cuda_Map_Class_Grid_pointers.size() != 2<<(N_dir-1) ) 
        { printf( "Grid pointers is supposed to have size %d.\n", 2<<(N_dir-1) ); fflush(stdout); exit(0); }


    // initialize cuda_class_grid
    for ( const auto & iter_grid_type : Array_Grid_types ) 
        { cuda_Map_Class_Grid_pointers.at(iter_grid_type)->cuda_Class_Grid_initialize ( Fwd_Specs.Map_Grid_pointers.at(iter_grid_type) ); }

    for ( const auto & iter_grid_type : Array_Grid_types ) 
        { cuda_Map_Class_Grid_pointers.at(iter_grid_type)->set_grid_pointers ( cuda_Map_Class_Grid_pointers ); }

    // NOTE: copy the parameters from hst to dev to start the simulation on device
    for ( const auto & iter_grid_type : Array_Grid_types ) 
    {
        cuda_Class_Grid_Base * cuda_class_grid = cuda_Map_Class_Grid_pointers.at(iter_grid_type);
        for ( int i_field = 0; i_field < cuda_class_grid->N_prmt; i_field++ )
        { 
            cuda_class_grid->copy_field_pitched ( "prmt" , i_field , "hst_to_dev" ); 
            
            if ( bool_energy )
                { cuda_class_grid->copy_field_pitched_enrg ( "enrg" , i_field , "hst_to_dev" ); }
        }
    }



printf("gpu parameter initialized.\n");

{
    // ns_type::cuda_precision dt_hst = static_cast<ns_type::cuda_precision> (ns_input::dt);
    // cudaMemcpyToSymbol ( ns_dev_var::dt, &dt_hst, sizeof(ns_type::cuda_precision) );
    // printf(" dt on host and device\n %16.15e\n", (double) ns_input::dt);
    // cuda_print_dt<<<1,1>>> ();
}
cudaDeviceSynchronize();


// ---- function body of forward_simulation is copied below so that 
//      we can experiment incrementally (one kernel at a time)
{
    Class_Grid & Grid_SNN = * ( Fwd_Specs.Map_Grid_pointers.at( {'N','N'} ) );
    Class_Grid & Grid_SMM = * ( Fwd_Specs.Map_Grid_pointers.at( {'M','M'} ) );

    Class_Grid & Grid_SMN  = * ( Fwd_Specs.Map_Grid_pointers.at( {'M','N'} ) );  // Pay attention here; we are using small 'x'
    Class_Grid & Grid_SNM  = * ( Fwd_Specs.Map_Grid_pointers.at( {'N','M'} ) );  // Pay attention here; we are using small 'y'

    auto & cuda_Class_Grid_SNN = grid_SNN;
    auto & cuda_Class_Grid_SMM = grid_SMM;

    auto & cuda_Class_Grid_SMN = grid_SMN;
    auto & cuda_Class_Grid_SNM = grid_SNM;


    // NOTE: It is very important that we don't forget the reference (&); otherwise, new objects will be
    //       instantiated while the interacting grid pointers they carried still point to the original ones.


    // ---- reset cpu memory to simulate again
    for ( const auto & iter_grid_type : Array_Grid_types ) 
        { for ( auto & iter : Fwd_Specs.Map_Grid_pointers.at(iter_grid_type)->Vec_soln ) { iter.set_constant(0.); } }


    // Allocate device memory to store RESULT
    for ( const auto & iter_grid_type : Array_Grid_types )  // go through the possible grid types
    {
        if ( Fwd_Specs.Map_grid_N_rcvs.at(iter_grid_type) > 0 )     // if number of receiver on this grid is larger than 0
        {
            auto & cuda_grid_rcv = cuda_Map_Class_Grid_pointers.at(iter_grid_type);
            for ( int i_rcv = 0; i_rcv < Fwd_Specs.Map_grid_N_rcvs.at(iter_grid_type); i_rcv++ )  // loop through the receivers
                { cuda_grid_rcv->Vec_RESULT_rcv.push_back ( cuda_run_time_matrix<ns_type::cuda_precision> { cuda_grid_rcv->N_soln , Nt } ); }
        }
    }


    for ( const auto & iter_grid_type : Array_Grid_types ) 
    {
        cuda_Class_Grid_Base * grid = cuda_Map_Class_Grid_pointers.at(iter_grid_type);



        // NOTE: stream_drvt is removed since reseting derivatives to zero is absorbed in their
        //       calculations - on gpu, the loop ordering does not affect the performance that 
        //       much. It's not as critical as on cpu, where the loop ordering can affect the
        //       performance of SIMD quite significantly. This has something to do with the 
        //       memory access pattern. On GPU, each thread access individual memory address, 
        //       the memory addresses accessed by a warp of threads need to be contiguous for
        //       good performance. On CPU, 1 thread needs to access contiguous memory addresses
        //       for them to be combined into SIMD operation.

        // NOTE: Because reseting derivative to zero is absorbed in their calculation, the 
        //       _secure_bdry_ function needs to be placed in the same stream as calculate_bdry
        //       and strictly afterwards (reset is done in calculate_bdry). stream_bx_L and
        //       stream_bx_R are therefore removed.

        // NOTE: The two functions _projection_ and _secure_boundary_ are now combined into a
        //       single function _secure_bdry_ and we indeed observe significant speed up. It 
        //       may be due to kernel launch overhead, or it may be due to the cache being "hot".
        //       However, when attempted to further combine the left and right sides of the 
        //       operations, the performance slowed down. (Curious why; maybe because this way
        //       we impose more and unnecessary dependence which makes it harder for the runtime 
        //       to schedule operations and use full capacity of the machine.) Therefore, it is 
        //       more likely to be due to the cache being hot.

        // NOTE: On CPU, replacing the single loop that use % operator with three loops that do 
        //       not use % operator leads to almost twice speedup. On GPU, we didn't observe the
        //       same thing, it actually slows the program down. Further, by removing the % from 
        //       the loop, i.e., performing incorrect simulation, the speedup is negligible. It 
        //       seems that GPU is not affected much by the % operator. On the other hand, when 
        //       we use three loops, it seems that the interior loop (the bulk part of the work) 
        //       has the following behavior - starting from zero and covering all points leads to
        //       faster program than covering only the interior points, i.e., more work version
        //       finishes sooner.
        //           Since the three-loop version does not show benefit, we remove them and the 
        //       associated streams and go light. 

        // NOTE: Events are removed since it seems to incur overhead when used inside for loop.
    }


printf("gpu stream created.\n");


    // NOTE: Using stream in combination with cudaDeviceSynchronize() seems to help; 
    //       using event doesn't seem to offer much benefit; there seems to be a claim 
    //       (look for cudaflow on youtube) that recording and waiting event can have 
    //       significant overhead, particularly when they are inside a loop; it's more 
    //       efficient to use graph, but cuda's graph API is not very easy to use - it
    //       can be cumbersome and error-prone to define the graph.
    //           We should probably save this for future, hence the event part of the 
    //       code is commented out above, and at the end of this file.


cudaDeviceSynchronize();

#ifndef CPSTFLAG
    constexpr int cpst_N = 0;  constexpr char cpst_S = '0';
#endif

#ifdef CPSTFLAG
#if CPSTFLAG == 0
    constexpr int cpst_N = 0;  constexpr char cpst_S = '0';
#elif CPSTFLAG == 3
    constexpr int cpst_N = 3;  constexpr char cpst_S = 'R';
#elif CPSTFLAG == 6
    constexpr int cpst_N = 6;  constexpr char cpst_S = 'R';
#endif
#endif
// [2024/04/03]
// NOTE: The "#ifdef CPSTFLAG" is needed because "#if CPSTFLAG == 0"
//       would evaluate to "true" if "CPSTFLAG" is undefined, which
//       would lead to double declaration of "cpst_N" and "cpst_S".


    // constexpr int  cpst_N =  0;  constexpr char cpst_S = '0';

    // constexpr int  cpst_N = -3;  constexpr char cpst_S = 'C';
    // constexpr int  cpst_N =  3;  constexpr char cpst_S = 'C';
    // constexpr int  cpst_N =  6;  constexpr char cpst_S = 'C';

    // constexpr int  cpst_N = -3;  constexpr char cpst_S = 'R';
    // constexpr int  cpst_N =  3;  constexpr char cpst_S = 'R';
    // constexpr int  cpst_N =  6;  constexpr char cpst_S = 'R';


    // Removed simulation loop 

cudaDeviceSynchronize();
printf("\n");


    // Removed outputs






} 

    // print_precision_type ();
    printf( "End of %s .\n", __FILE__ ); fflush(stdout);

    return 0;
}
