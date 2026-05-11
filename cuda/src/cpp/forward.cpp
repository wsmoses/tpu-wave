#include "forward.hpp"

using ns_input::Nt;

using ns_input::dx;
using ns_input::dt;

using ns_input::bool_visual;

void Class_Forward_Specs::process_src_locations ( struct_src_input & src_input )
{
    src_forward.src_index = src_input.src_index; // Required to avoid out_of_range in main loop
}

void Class_Forward_Specs::process_rcv_locations ( std::vector< struct_rcv_input > & Vec_Rcv_Input ) 
{
    constexpr int N_dir = ns_forward::N_dir;

    std::array<char, N_dir> grid_type = {'M', 'M'};

    // Just call .at() without the loop
    auto& record_rcv = Map_grid_record_rcv.at(grid_type);
    (void)record_rcv; // Suppress unused variable warning
}

void Class_Forward_Specs::output_solution_record ( std::string folder_name ) {}
void Class_Forward_Specs::output_energy ( std::string folder_name ) {}
void Class_Forward_Specs::calculate_data_misfit () {}