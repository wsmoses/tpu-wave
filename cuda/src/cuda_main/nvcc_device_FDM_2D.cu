#include "forward.hpp"
#include "inverse.hpp"

#include "grid.cuh"
#include "namespace_device_variable.cuh"  /* probably dummy, only the #include "namespace_type.cuh" inside it is meaningful */

int main(int argc, char* argv[]) 
{
    Class_Forward_Specs Fwd_Specs;
    Class_Inverse_Specs Inv_Specs;
    
    constexpr auto Array_Grid_types = ns_forward::define_Array_Grid_types ();
    std::map< int , std::vector< struct_rcv_input > > & Map_Vec_Rcv_Input = Inv_Specs.Map_Vec_Rcv_Input;

    struct_src_input iter_vec { 1 , {'N','N'} , {0,1,0,1} };
    Map_Vec_Rcv_Input[1] = { struct_rcv_input { 1 , 1 , {'N','N'} , {0,1,0,1} } };

    //Inv_Specs.data_misfit = 0.;  // (re)set the aggregated data misfit for all sources to zero before 
                                 // enter the loop that goes through the sources and lauches simulations 

    for (int i=0; i<1; i++) {
        int fwd_src_index = 1;//Fwd_Specs.src_forward.src_index;

        for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_record_rcv        [grid_type] = {}; }

        Fwd_Specs.process_rcv_locations ( Map_Vec_Rcv_Input.at( fwd_src_index ) );
    }

    return 0;
}
