#include "forward.hpp"

using ns_input::Nt;

using ns_input::dx;
using ns_input::dt;

using ns_input::bool_visual;

void Class_Forward_Specs::process_src_locations ( struct_src_input & src_input )
{
    constexpr int N_dir = ns_forward::N_dir;

    std::array<char, N_dir> & grid_type = src_input.src_grid_type;
    Class_Grid * grid_src = Map_Grid_pointers.at(grid_type);

    std::array<long, N_dir> num_location {}; num_location.fill(-1);
    std::array<long, N_dir> den_location {}; den_location.fill(-1);
    for ( int i_dir = 0; i_dir < N_dir; i_dir++ ) 
    {
        num_location[i_dir] = src_input.src_location[i_dir*2  ];
        den_location[i_dir] = src_input.src_location[i_dir*2+1];    
    }

    std::array<long, N_dir> index_src {}; index_src.fill(-1);

    for ( int i_dir = 0; i_dir < N_dir; i_dir++ )
    {
        if ( grid_type[i_dir] == 'N' )
        {
            long num_src = num_location[i_dir] * ns_input::den_dx_soln;
            long den_src = den_location[i_dir] * ns_input::num_dx_soln;
            index_src[i_dir] = num_src / den_src;
        }
        else if ( grid_type[i_dir] == 'M' )
        {
            long num_src = 2 * num_location[i_dir] * ns_input::den_dx_soln - den_location[i_dir] * ns_input::num_dx_soln;
            long den_src = 2 * den_location[i_dir] * ns_input::num_dx_soln;
            index_src[i_dir] = num_src / den_src;
        }
    }
    
    src_forward.is_src = true;
    src_forward.src_index = src_input.src_index;
    src_forward.grid_type = src_input.src_grid_type;
    src_forward.ix_src = index_src[0] - grid_src->G_ix_bgn;
    src_forward.iy_src = index_src[1] - grid_src->G_iy_bgn;
    src_forward.I_SRC  = src_forward.ix_src * grid_src->G_size_y + src_forward.iy_src;
    src_forward.c_source_type = grid_src->grid_name.at(0);

    printf( "\n ---- Found source of type %c at (%3d %3d - %d) on grid (%c %c) of sizes (%3d %3d)\n\n", 
            src_forward.c_source_type,
            src_forward.ix_src, src_forward.iy_src, src_forward.I_SRC, 
            grid_src->G_type_x,  grid_src->G_type_y,  
            grid_src->G_size_x,  grid_src->G_size_y );
    fflush(stdout);
}

void Class_Forward_Specs::process_rcv_locations ( std::vector< struct_rcv_input > & Vec_Rcv_Input ) 
{
    constexpr int N_dir = ns_forward::N_dir;

    for ( const auto & rcv_input : Vec_Rcv_Input ) 
    {
        std::array<char, N_dir> grid_type = rcv_input.rcv_grid_type;

        if ( true )
        {
            Map_grid_N_rcvs.at(grid_type) += 1;
        }
    }

    for ( const auto & iter_map : Map_Grid_pointers ) 
    {
        std::array<char, N_dir> grid_type = iter_map.first;

        for ( auto& record_rcv : Map_grid_record_rcv.at(grid_type) ) 
            { /* dummy */ }
    }
}

void Class_Forward_Specs::output_solution_record ( std::string folder_name ) {}
void Class_Forward_Specs::output_energy ( std::string folder_name ) {}
void Class_Forward_Specs::calculate_data_misfit () {}