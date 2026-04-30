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

        cudaStreamCreate( & grid->stream_dx_I );
        cudaStreamCreate( & grid->stream_dx_L );
        cudaStreamCreate( & grid->stream_dx_R );

        cudaStreamCreate( & grid->stream_dy );

        cudaStreamCreate( & grid->stream_soln );

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


    for (int it=0; it<Nt; it++) 
    {

cudaDeviceSynchronize(); // debug

        if ( bool_energy ) { cudaStreamSynchronize ( cuda_Class_Grid_SMM.stream_soln ); ns_input::Record_E_p0.at(it) += cuda_Class_Grid_SMM.energy_calculation (); }
        if ( bool_energy ) { cudaStreamSynchronize ( cuda_Class_Grid_SNN.stream_soln ); ns_input::Record_E_p0.at(it) += cuda_Class_Grid_SNN.energy_calculation (); }

cudaDeviceSynchronize(); // debug

        // ---- V
        // NOTE: reset the derivatives is absorbed in their calculation.


        // below is important: Sxy should finish updating a lot quicker than SNN; we should dispatch those work that only requires Sxy
        //                     as soon as Sxy is ready, rather than waiting for SNN        


        // NOTE: Below is important: the update of Sxy and SNN may start around the 
        //       same time, but Sxy will finish sooner; we should dispatch those 
        //       work that requires only Sxy as soon as Sxy is ready, and not to 
        //       wait for the update of SNN to finish - this leads to significant 
        //       improvement in performance. (Similar ordering does not show much 
        //       benefit for the update of Vx and Vy, presumbly because they took
        //       roughly the same time.)

        cudaStreamSynchronize ( cuda_Class_Grid_SNN.stream_soln );
        // NOTE: Pay attention to the XY and xy distinction. When Sxy is ready, 
        //       the y derivative of NM grid can be calculated, which is where 
        //       Vy is, not Vx.

cudaDeviceSynchronize(); // debug

        // y derivative of Vy
        cuda_Class_Grid_SNM.kernel_launch_cuda_periodic_y_modulo <false, 'N', 'N'> ();

cudaDeviceSynchronize(); // debug

        // x derivative of Vx
        {
            cuda_Class_Grid_SMN.kernel_launch_cuda_interior_x ();

            cuda_Class_Grid_SMN.kernel_launch_cuda_boundary_x ( 'L' );
            cuda_Class_Grid_SMN.kernel_launch_cuda_boundary_x ( 'R' );

            cuda_Class_Grid_SMN.kernel_launch_cuda_secure_bdry_x ( 'L' );
            cuda_Class_Grid_SMN.kernel_launch_cuda_secure_bdry_x ( 'R' );
        }

cudaDeviceSynchronize(); // debug

        cudaStreamSynchronize ( cuda_Class_Grid_SMM.stream_soln );

        // x derivative of Vy
        {
            cuda_Class_Grid_SNM.kernel_launch_cuda_interior_x ();
            
            cuda_Class_Grid_SNM.kernel_launch_cuda_boundary_x ( 'L' );
            cuda_Class_Grid_SNM.kernel_launch_cuda_boundary_x ( 'R' );
    
            cuda_Class_Grid_SNM.kernel_launch_cuda_secure_bdry_x ( 'L' );
            cuda_Class_Grid_SNM.kernel_launch_cuda_secure_bdry_x ( 'R' );
        }

cudaDeviceSynchronize(); // debug
        
        // y derivative of Vx
        cuda_Class_Grid_SMN.kernel_launch_cuda_periodic_y_modulo <false, 'M', 'M'> ();

cudaDeviceSynchronize(); // debug

        // update
        cudaStreamSynchronize( cuda_Class_Grid_SNM.stream_dx_I );
        cudaStreamSynchronize( cuda_Class_Grid_SNM.stream_dx_L );
        cudaStreamSynchronize( cuda_Class_Grid_SNM.stream_dx_R );
        cudaStreamSynchronize( cuda_Class_Grid_SNM.stream_dy   );

        cuda_Class_Grid_SNM.kernel_launch_cuda_update <cpst_N , cpst_S> ();
    
cudaDeviceSynchronize(); // debug

        cudaStreamSynchronize( cuda_Class_Grid_SMN.stream_dx_I );
        cudaStreamSynchronize( cuda_Class_Grid_SMN.stream_dx_L );
        cudaStreamSynchronize( cuda_Class_Grid_SMN.stream_dx_R );
        cudaStreamSynchronize( cuda_Class_Grid_SMN.stream_dy   );

        cuda_Class_Grid_SMN.kernel_launch_cuda_update <cpst_N , cpst_S> ();

cudaDeviceSynchronize(); // debug

        // Apply source
        if ( Fwd_Specs.src_forward.is_src ) 
        {   
            if ( Fwd_Specs.src_forward.c_source_type == 'V' ) // if source is on a velocity grid
            {
                double t, A;
                t = it * (double) dt - (double) time_delay;       // No kidding, 1/2 must have given zero
                A = (1-2*(double) M_PI*(double) M_PI*(double) central_f*(double) central_f*(double) t*(double) t)
                  * exp(-(double) M_PI*(double) M_PI*(double) central_f*(double) central_f*(double) t*(double) t);

                auto const & cuda_grid_src = cuda_Map_Class_Grid_pointers.at( Fwd_Specs.src_forward.grid_type );

                int ind = Fwd_Specs.src_forward.ix_src * cuda_grid_src->Ly_pad
                        + Fwd_Specs.src_forward.iy_src;

                double increment = A * dt / (dx * dx) / ns_input::source_scaling;

                for ( int i_field = 0; i_field < cuda_grid_src->N_soln; i_field++ )
                {
                    ns_type::cuda_precision * S = cuda_grid_src->Vec_soln.at(i_field).ptr;
                    cuda_apply_source <<< 1 , 1 , 0 , cuda_grid_src->stream_soln >>> ( S , ind , increment );
                    // if ( cpst_N == 0 )
                    // {
                    //     cuda_apply_source <<< 1 , 1 , 0 , cuda_grid_src->stream_soln >>> ( S , ind , increment );
                    // }
                    // else
                    // {
                    //     ns_type::cuda_precision * R = cuda_grid_src->src_cpst.ptr;
                    //     cuda_apply_source <<< 1 , 1 , 0 , cuda_grid_src->stream_soln >>> ( S , ind , R , i_field, increment );
                    // }
                    // [2024/03/26]
                    // NOTE: I think the above branch is for the purpose of testing if applying compensation to source
                    //       application would make a difference. (The cuda_apply_source function is in grid.cu file.)
                }
            }
        }

cudaDeviceSynchronize(); // debug

        if ( bool_energy ) { cudaStreamSynchronize ( cuda_Class_Grid_SMN .stream_soln ); ns_input::Record_E_k.at(it) += cuda_Class_Grid_SMN.energy_calculation (); }
        if ( bool_energy ) { cudaStreamSynchronize ( cuda_Class_Grid_SNM .stream_soln ); ns_input::Record_E_k.at(it) += cuda_Class_Grid_SNM.energy_calculation (); }

cudaDeviceSynchronize(); // debug

        // ---- S
        // NOTE: reset the derivatives is absorbed in their calculation.

        cudaStreamSynchronize ( cuda_Class_Grid_SNM .stream_soln );

        // y derivative of SNN
        cuda_Class_Grid_SNN.kernel_launch_cuda_periodic_y_modulo <false, 'N', 'M'> ();

cudaDeviceSynchronize(); // debug

        // x derivative of SMM
        {
            cuda_Class_Grid_SMM.kernel_launch_cuda_interior_x <true> ();
            cuda_Class_Grid_SMM.kernel_launch_cuda_boundary_x <true> ( 'L' );
            cuda_Class_Grid_SMM.kernel_launch_cuda_boundary_x <true> ( 'R' );
        }

cudaDeviceSynchronize(); // debug

        cudaStreamSynchronize ( cuda_Class_Grid_SMN .stream_soln );

        // x derivative of SNN
        {
            cuda_Class_Grid_SNN.kernel_launch_cuda_interior_x ();
            cuda_Class_Grid_SNN.kernel_launch_cuda_boundary_x ( 'L' );
            cuda_Class_Grid_SNN.kernel_launch_cuda_boundary_x ( 'R' );
        }
        // y derivative of SMM
        cuda_Class_Grid_SMM.kernel_launch_cuda_periodic_y_modulo <true, 'M', 'N'> ();

cudaDeviceSynchronize(); // debug        

        // update
        cudaStreamSynchronize( cuda_Class_Grid_SNN.stream_dx_I );
        cudaStreamSynchronize( cuda_Class_Grid_SNN.stream_dx_L );
        cudaStreamSynchronize( cuda_Class_Grid_SNN.stream_dx_R );
        cudaStreamSynchronize( cuda_Class_Grid_SNN.stream_dy );

        cuda_Class_Grid_SNN.kernel_launch_cuda_update <cpst_N , cpst_S> ();

cudaDeviceSynchronize(); // debug

        cudaStreamSynchronize( cuda_Class_Grid_SMM.stream_dx_I );
        cudaStreamSynchronize( cuda_Class_Grid_SMM.stream_dx_L );
        cudaStreamSynchronize( cuda_Class_Grid_SMM.stream_dx_R );
        cudaStreamSynchronize( cuda_Class_Grid_SMM.stream_dy );

        cuda_Class_Grid_SMM.kernel_launch_cuda_update <cpst_N , cpst_S> ();
        
cudaDeviceSynchronize(); // debug

        // Apply source
        if ( Fwd_Specs.src_forward.is_src ) 
        {   
            if ( Fwd_Specs.src_forward.c_source_type == 'S' ) // if source is on a stress grid
            {
                double t, A;
                t = (it+1./2.)*(double) dt - (double) time_delay;       // No kidding, 1/2 must have given zero
                A = (1-2*(double) M_PI*(double) M_PI*(double) central_f*(double) central_f*(double) t*(double) t)
                  * exp(-(double) M_PI*(double) M_PI*(double) central_f*(double) central_f*(double) t*(double) t);

                auto const & cuda_grid_src = cuda_Map_Class_Grid_pointers.at( Fwd_Specs.src_forward.grid_type );

                int ind = Fwd_Specs.src_forward.ix_src * cuda_grid_src->Ly_pad
                        + Fwd_Specs.src_forward.iy_src;

                double increment = A * dt / (dx * dx) / ns_input::source_scaling;

                for ( int i_field = 0; i_field < cuda_grid_src->N_soln; i_field++ )
                { 
                    ns_type::cuda_precision * S = cuda_grid_src->Vec_soln.at(i_field).ptr;
                    cuda_apply_source <<< 1 , 1 , 0 , cuda_grid_src->stream_soln >>> ( S , ind , increment );
                    // if ( cpst_N == 0 )
                    // {
                    //     cuda_apply_source <<< 1 , 1 , 0 , cuda_grid_src->stream_soln >>> ( S , ind , increment );
                    // }
                    // else
                    // {
                    //     ns_type::cuda_precision * R = cuda_grid_src->src_cpst.ptr;
                    //     cuda_apply_source <<< 1 , 1 , 0 , cuda_grid_src->stream_soln >>> ( S , ind , R , i_field, increment );
                    // }
                    // [2024/03/26]
                    // NOTE: I think the above branch is for the purpose of testing if applying compensation to source
                    //       application would make a difference. (The cuda_apply_source function is in grid.cu file.)
                }
            }
        }


cudaDeviceSynchronize(); // debug


// NOTE: most of the above code for applying source can be taken outside the time loop


        if ( bool_energy ) { cudaStreamSynchronize ( cuda_Class_Grid_SMM.stream_soln ); ns_input::Record_E_p1.at(it) += cuda_Class_Grid_SMM.energy_calculation (); }
        if ( bool_energy ) { cudaStreamSynchronize ( cuda_Class_Grid_SNN.stream_soln ); ns_input::Record_E_p1.at(it) += cuda_Class_Grid_SNN.energy_calculation (); }


cudaDeviceSynchronize(); // debug

        // Store solution RESULT
        for ( const auto & iter_map : Fwd_Specs.Map_grid_N_rcvs )  // go through the possible grid types
        {
            auto & grid_type = iter_map.first;

            if ( Fwd_Specs.Map_grid_N_rcvs.at(grid_type) > 0 )     // if number of receiver on this grid is larger than 0
            {
                auto & cuda_grid_rcv = cuda_Map_Class_Grid_pointers.at(grid_type);

                auto & Vec_struct_rcv = Fwd_Specs.Map_grid_struct_rcv_forward.at(grid_type);
                auto & Vec_RESULT_rcv = cuda_grid_rcv->Vec_RESULT_rcv;

                for ( int i_rcv = 0; i_rcv < Fwd_Specs.Map_grid_N_rcvs.at(grid_type); i_rcv++ )  // loop through the receivers
                {
                    int I_RCV              = Vec_struct_rcv.at(i_rcv).ix_rcv * cuda_grid_rcv->Ly_pad
                                           + Vec_struct_rcv.at(i_rcv).iy_rcv;

                    auto & Solution_RESULT = Vec_RESULT_rcv.at(i_rcv);

                    for ( int i_field = 0; i_field < cuda_grid_rcv->N_soln; i_field++ )    // loop through the fields on this grid
                    { 
                        ns_type::cuda_precision * R = Solution_RESULT.ptr + i_field * Nt + it;
                        ns_type::cuda_precision * S = cuda_grid_rcv->Vec_soln.at(i_field).ptr;
                        cuda_record_soln <<< 1 , 1 , 0 , cuda_grid_rcv->stream_soln >>> ( R , S , I_RCV );
                    }
                }
            }
        }

cudaDeviceSynchronize(); // debug

        if ( (it % 1000) == 0 )
        {
        	printf( "\n" ); fflush(stdout);
    		auto const & cuda_grid_src = cuda_Map_Class_Grid_pointers.at( Fwd_Specs.src_forward.grid_type );
        	int ind = Fwd_Specs.src_forward.ix_src * cuda_grid_src->Ly_pad
            	    + Fwd_Specs.src_forward.iy_src;

        	for ( int i_field = 0; i_field < cuda_grid_src->N_soln; i_field++ )
            	{ cuda_print_soln <<< 1 , 1 >>> ( cuda_grid_src->Vec_soln.at(i_field).ptr , ind , it ); }
        }
        // [2024/03/27]
        // NOTE: Made the above changes on the printing out the solution at the src location. Before, we always print out
        //       Sxx and Syy - this can lead to mismatch in nvcc_host_FDM_2D.exe and nvcc_device_FDM_2D.exe when the src
        //       is not applied on the {Sxx;Syy} grid. The reason is that the device code uses padding.

    }  // for (int it=0; it<Nt; it++) 

cudaDeviceSynchronize();
printf("\n");


    // copy RESULT to host memory
    for ( const auto & iter_grid_type : Array_Grid_types )  // go through the possible grid types
    {
        if ( Fwd_Specs.Map_grid_N_rcvs.at(iter_grid_type) > 0 )     // if number of receiver on this grid is larger than 0
        {
            auto &      grid_rcv =  Fwd_Specs.Map_Grid_pointers.at(iter_grid_type);
            auto & cuda_grid_rcv = cuda_Map_Class_Grid_pointers.at(iter_grid_type);
            for ( int i_rcv = 0; i_rcv < Fwd_Specs.Map_grid_N_rcvs.at(iter_grid_type); i_rcv++ )  // loop through the receivers
            { 
                cudaMemcpy ( Fwd_Specs.Map_grid_RESULT_rcv.at(iter_grid_type).at(i_rcv).ptr ,
                             cuda_grid_rcv->Vec_RESULT_rcv.at(i_rcv).ptr , 
                             cuda_grid_rcv->N_soln * Nt * sizeof(ns_type::cuda_precision) , 
                             cudaMemcpyDeviceToHost );
            }
        }
    }

    // ---- output the results
    char dt_folder[22];  // 2 (?.) + 15 + 4 (e-0?) + 1 ('\0')
    sprintf( dt_folder, "dt_%16.15e", (double) ns_input::dt );
    dt_folder[4] = 'p';

    std::string str_precision = "Unrecognized type";
    if ( std::is_same_v < ns_type::cuda_precision , double        > ) { str_precision = "fp64"; }
    if ( std::is_same_v < ns_type::cuda_precision , float         > ) { str_precision = "fp32"; }
    if ( std::is_same_v < ns_type::cuda_precision , _Float16      > ) { str_precision = "fp16"; } // this would compile?
    if ( std::is_same_v < ns_type::cuda_precision , __half        > ) { str_precision = "fp16"; }
    if ( std::is_same_v < ns_type::cuda_precision , __nv_bfloat16 > ) { str_precision = "bf16"; }
    printf( "\nPrecision type: %s .\n", str_precision.c_str() );

    std::string folder_name = "../../result/elastic/" + medium_name + "/device" 
    						+ "/grid_size" + "_" + std::to_string(ns_input::num_dx_soln)
                                           + "_" + std::to_string(ns_input::den_dx_soln)
                            + "/Nt_" + std::to_string(Nt) 
                            + "/" + str_precision
                            + "/" + std::string(dt_folder)
                            + "/" + std::to_string(cpst_N) + "_" + cpst_S;

    std::filesystem::create_directories( folder_name );
    std::cout << "Results will be stored in folder: \n" << folder_name << std::endl;
    // [2023/09/20] NOTE: create_directories () can take both char* and std::string.

	std::cout << "Grid related parameters were read from: \n" << input_file_name << std::endl;

    // ---- output the solution  
    Fwd_Specs.output_solution_record ( folder_name );

    // ---- record the energy if requested
    if ( bool_energy ) { Fwd_Specs.output_energy ( folder_name ); }


    // NOTE: copy the soln from dev to hst for the final snapshot
    for ( const auto & iter_grid_type : Array_Grid_types ) 
    {
        cuda_Class_Grid_Base * cuda_class_grid = cuda_Map_Class_Grid_pointers.at(iter_grid_type);
        for ( int i_field = 0; i_field < cuda_class_grid->N_soln; i_field++ )
            { cuda_class_grid->copy_field_pitched ( "soln" , i_field , "dev_to_hst" ); }
    }

    // ---- output the final snapshot
    for ( const auto & iter_grid_type : Array_Grid_types ) 
    {
        // std::array<char, ns_forward::N_dir> physical_grid_type;
        // physical_grid_type.at(0) = iter_grid_type.at( ns_forward::XY_to_01 ('X') );
        // physical_grid_type.at(1) = iter_grid_type.at( ns_forward::XY_to_01 ('Y') );

        Class_Grid * grid = Fwd_Specs.Map_Grid_pointers.at(iter_grid_type);

        for ( int i_field = 0; i_field < grid->N_soln; i_field++ )
        { 
            std::string file_name = folder_name + "/Snapshot"
                                  + "_" + iter_grid_type.at(1) + iter_grid_type.at(0)    // grid type
                                  + "_" + grid->grid_name
                                  + "_" + std::to_string( i_field );                     // which field on this grid

            auto & soln = grid->Vec_soln.at(i_field);
            FILE * fp = fopen( ( file_name + ".txt" ).c_str(), "w" );
            for ( int i = 0; i < grid->G_size_x * grid->G_size_y; i++ ) 
            {
                fprintf( fp, "% 16.15e\n", ns_input::source_scaling * (double) soln.at(i) );             
            }
            fclose(fp);
        }
    }




    for ( const auto & iter_grid_type : Array_Grid_types ) 
    {
        cuda_Class_Grid_Base * grid = cuda_Map_Class_Grid_pointers.at(iter_grid_type);

        cudaStreamDestroy( grid->stream_dx_I );
        cudaStreamDestroy( grid->stream_dx_L );
        cudaStreamDestroy( grid->stream_dx_R );

        cudaStreamDestroy( grid->stream_dy );

        cudaStreamDestroy( grid->stream_soln );
    }

} 

    // print_precision_type ();
    printf( "End of %s .\n", __FILE__ ); fflush(stdout);

    return 0;
}
