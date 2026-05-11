#include "forward.hpp"

int main() {
    Class_Forward_Specs Fwd_Specs;
    
    // We need 4 grids to map to the 4 combinations of N and M
    std::array<Class_Grid, 4> Grids;
    
    std::array<std::array<char, 2>, 4> Array_Grid_types = {{
        {'M', 'M'}, {'M', 'N'}, {'N', 'M'}, {'N', 'N'}
    }};

    // Initialize Map_Grid_pointers
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types[0]] = &Grids[0];
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types[1]] = &Grids[1];
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types[2]] = &Grids[2];
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types[3]] = &Grids[3];

    // Initialize the maps as in the original code
    for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_N_rcvs            [grid_type] = 0;  }
    for ( const auto & grid_type : Array_Grid_types ) { Fwd_Specs.Map_grid_record_rcv        [grid_type] = {}; }

    // Create a dummy receiver input to trigger the first loop
    std::vector<struct_rcv_input> Vec_Rcv_Input;
    struct_rcv_input rcv;
    rcv.rcv_grid_type = {'N', 'M'};
    Vec_Rcv_Input.push_back(rcv);

    printf("Starting process_rcv_locations...\n"); fflush(stdout);
    Fwd_Specs.process_rcv_locations(Vec_Rcv_Input);
    printf("Finished process_rcv_locations successfully!\n"); fflush(stdout);

    return 0;
}
