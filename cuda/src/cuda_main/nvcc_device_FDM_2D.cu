#include "forward.hpp"
#include <stdio.h>
#include <vector>

int main(int argc, char* argv[]) 
{
    Class_Forward_Specs Fwd_Specs;
    using ns_forward::N_dir;
    std::array< Class_Grid , 2<<(N_dir-1) > Grids;

    constexpr auto Array_Grid_types = ns_forward::define_Array_Grid_types ();

    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(0)] = &Grids.at(0);
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(1)] = &Grids.at(1);
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(2)] = &Grids.at(2);
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types.at(3)] = &Grids.at(3);

    printf("Array_Grid_types size: %lu\n", Array_Grid_types.size());
    for (auto const& type : Array_Grid_types) {
        printf("Type: %c %c\n", type[0], type[1]);
    }

    for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_N_rcvs            [grid_type] = 0;  }
    for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_struct_rcv_forward[grid_type] = {}; }
    for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_record_rcv        [grid_type] = {}; }
    for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_RESULT_rcv        [grid_type] = {}; }
    for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_misfit_rcv        [grid_type] = {}; }

    std::vector< struct_rcv_input > dummy_rcv;
    Fwd_Specs.process_rcv_locations ( dummy_rcv );

    return 0;
}
