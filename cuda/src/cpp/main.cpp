#include "forward.hpp"

// Define the extern variables needed by set_grid_parameters
namespace namespace_input {
    long Nx_soln = 600;
    long Mx_soln = 600;
    long Ny_soln = 600;
    long My_soln = 600;
    
    // Also need these for allocate_memory in set_grid_parameters
    run_time_vector<double> Record_E_k;
    run_time_vector<double> Record_E_p0;
    run_time_vector<double> Record_E_p1;
}

int main() {
    Class_Forward_Specs Fwd_Specs;
    
    std::array<Class_Grid, 4> Grids;
    
    constexpr auto Array_Grid_types = ns_forward::define_Array_Grid_types();

    // Initialize Map_Grid_pointers
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types[0]] = &Grids[0];
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types[1]] = &Grids[1];
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types[2]] = &Grids[2];
    Fwd_Specs.Map_Grid_pointers[Array_Grid_types[3]] = &Grids[3];

    // Call set_grid_parameters as in the original code
    for ( const auto & grid_type : Array_Grid_types ) { 
        Fwd_Specs.Map_Grid_pointers.at(grid_type)->set_grid_parameters ( grid_type , false ); 
    }

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
