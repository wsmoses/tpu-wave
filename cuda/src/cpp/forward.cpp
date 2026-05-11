#include "forward.hpp"

void Class_Forward_Specs::process_src_locations ( struct_src_input & src_input ) {}

void Class_Forward_Specs::process_rcv_locations ( std::vector< struct_rcv_input > & Vec_Rcv_Input ) 
{
    for ( const auto & rcv_input : Vec_Rcv_Input ) 
    {
        std::array<char, 2> grid_type = rcv_input.rcv_grid_type;

        if ( true )
        {
            Map_grid_N_rcvs.at(grid_type) += 1;
        }
    }

    for ( const auto & iter_map : Map_Grid_pointers ) 
    {
        std::array<char, 2> grid_type = iter_map.first;

        for ( auto& record_rcv : Map_grid_record_rcv.at(grid_type) ) 
            { /* dummy */ }
    }
}

void Class_Forward_Specs::output_solution_record ( std::string folder_name ) {}
void Class_Forward_Specs::output_energy ( std::string folder_name ) {}
void Class_Forward_Specs::calculate_data_misfit () {}