#include "forward.hpp"

void Class_Forward_Specs::process_src_locations ( struct_src_input & src_input )
{
    src_forward.src_index = src_input.src_index; // Required to avoid out_of_range in main loop
}

void Class_Forward_Specs::process_rcv_locations ( std::vector< struct_rcv_input > & Vec_Rcv_Input ) 
{
    // Call .at() with a temporary key
    auto& record_rcv = Map_grid_record_rcv.at({'M', 'M'});
    (void)record_rcv; // Suppress unused variable warning
}

void Class_Forward_Specs::output_solution_record ( std::string folder_name ) {}
void Class_Forward_Specs::output_energy ( std::string folder_name ) {}
void Class_Forward_Specs::calculate_data_misfit () {}