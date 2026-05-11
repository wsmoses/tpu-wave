#include "forward.hpp"
#include "inverse.hpp"

#include "grid.cuh"
#include "namespace_device_variable.cuh"  /* probably dummy, only the #include "namespace_type.cuh" inside it is meaningful */

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

    // Removed input processing


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



    // Removed physical parameters setup


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

		    struct_src_input iter_vec { 1 , {'N','N'} , {0,1,0,1} };
    Map_Vec_Rcv_Input[1] = { struct_rcv_input { 1 , 1 , {'N','N'} , {0,1,0,1} } };

    Inv_Specs.data_misfit = 0.;  // (re)set the aggregated data misfit for all sources to zero before 
                                 // enter the loop that goes through the sources and lauches simulations 

    for (int i=0; i<1; i++) {
        // ---- Process the source and receiver location 

        //Fwd_Specs.process_src_locations ( iter_vec );
        int fwd_src_index = 1;//Fwd_Specs.src_forward.src_index;

        printf( "    Processing receiver locations for source %d.\n", fwd_src_index );


        Fwd_Specs.process_rcv_locations ( Map_Vec_Rcv_Input.at( fwd_src_index ) );
    }

    return 0;
}
