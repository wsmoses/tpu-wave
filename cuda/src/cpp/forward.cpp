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

    // inputted 'physical' source location
    std::array<long, N_dir> num_location {}; num_location.fill(-1);
    std::array<long, N_dir> den_location {}; den_location.fill(-1);
    for ( int i_dir = 0; i_dir < N_dir; i_dir++ ) 
    {
        num_location[i_dir] = src_input.src_location[i_dir*2  ];
        den_location[i_dir] = src_input.src_location[i_dir*2+1];    
    }

    // global source index
    std::array<long, N_dir> index_src {}; index_src.fill(-1); // "index" is really "location" (or grid index)

    // Convert the 'physical' location to global indices
    for ( int i_dir = 0; i_dir < N_dir; i_dir++ )
    {
        if ( grid_type[i_dir] == 'N' )
        {
            // check if the 'physical' location can be converted to integer indices on this grid
            long num_src = num_location[i_dir] * ns_input::den_dx_soln;
            long den_src = den_location[i_dir] * ns_input::num_dx_soln;
            if ( den_src == 0 ) { printf("den_src should not be zero.\n"); fflush(stdout); exit(0); }
            if ( num_src % den_src != 0 )
                { printf("Cannot convert physical location to integer index on direction %d.\n", i_dir); fflush(stdout); exit(0); }
            index_src[i_dir] = num_src / den_src;
        }
        else if ( grid_type[i_dir] == 'M' )
        {
            // check if the 'physical' location can be converted to integer indices on this grid
            long num_src = 2 * num_location[i_dir] * ns_input::den_dx_soln - den_location[i_dir] * ns_input::num_dx_soln;
            long den_src = 2 * den_location[i_dir] * ns_input::num_dx_soln;
            if ( den_src == 0 ) { printf("den_src should not be zero.\n"); fflush(stdout); exit(0); }
            if ( num_src % den_src != 0 )
                { printf("Cannot convert physical location to integer index on direction %d.\n", i_dir); fflush(stdout); exit(0); }
            index_src[i_dir] = num_src / den_src;
        }
        else
            { printf("Grid type unrecognized %c.\n", grid_type[i_dir]); fflush(stdout); exit(0); }
    }
    
    // the worker double-checks if it is indeed responsible for this source
    if ( grid_src->G_ix_bgn <= index_src[0] && index_src[0] < grid_src->G_ix_end && 
         grid_src->G_iy_bgn <= index_src[1] && index_src[1] < grid_src->G_iy_end )
    {
        src_forward.is_src = true;

        src_forward.src_index = src_input.src_index;
        src_forward.grid_type = src_input.src_grid_type;

        src_forward.ix_src = index_src[0] - grid_src->G_ix_bgn;
        src_forward.iy_src = index_src[1] - grid_src->G_iy_bgn;

        src_forward.I_SRC  = src_forward.ix_src * grid_src->G_size_y 
                           + src_forward.iy_src;

        src_forward.c_source_type = grid_src->grid_name.at(0);     // grid_name is an std::string, starting with either 'S' or 'V';
        if ( src_forward.c_source_type != 'S' && src_forward.c_source_type != 'V' )
            { printf( "Source grid type unrecognized %c.\n", src_forward.c_source_type ); fflush(stdout); exit(0); }

        printf( "\n ---- Found source of type %c at (%3d %3d - %d) on grid (%c %c) of sizes (%3d %3d)\n\n", 
                src_forward.c_source_type,
                src_forward.ix_src, src_forward.iy_src, src_forward.I_SRC, 
                grid_src->G_type_x,  grid_src->G_type_y,  
                grid_src->G_size_x,  grid_src->G_size_y );
        fflush(stdout);
    }
    else 
        { 
            printf( "\n ---- Found source of type %c at (%3d %3d - %d) on grid (%c %c) of sizes (%3d %3d)\n\n", 
                src_forward.c_source_type,
                src_forward.ix_src, src_forward.iy_src, src_forward.I_SRC, 
                grid_src->G_type_x,  grid_src->G_type_y,  
                grid_src->G_size_x,  grid_src->G_size_y );
            fflush(stdout);

            printf("Source outside of simulation region %s %d.\n", __FILE__, __LINE__); fflush(stdout); exit(0); 
        }

    // test the special cases that we cannot handle yet
    if ( ( index_src[0] == grid_src->G_ix_bgn || index_src[0] == grid_src->G_ix_end ) && src_forward.grid_type[0] == 'N' ) 
        { printf("Source is on weak boundary in the x direction. Is this okay?\n"); fflush(stdout); exit(0); }
    if ( ( index_src[1] == grid_src->G_iy_bgn || index_src[1] == grid_src->G_iy_end ) && src_forward.grid_type[1] == 'N' ) 
        { printf("Source is on weak boundary in the y direction. Is this okay?\n"); fflush(stdout); } // exit(0); }

} // Class_Forward_Specs::process_src_locations ()



void Class_Forward_Specs::process_rcv_locations ( std::vector< struct_rcv_input > & Vec_Rcv_Input ) 
{
    constexpr int N_dir = ns_forward::N_dir;
    
    for ( const auto & rcv_input : Vec_Rcv_Input ) 
    {
        std::array<char, N_dir> grid_type = rcv_input.rcv_grid_type;
        Class_Grid * grid_rcv = Map_Grid_pointers.at(grid_type);

        // inputted 'physical' source location
        std::array<long, N_dir> num_location {}; num_location.fill(-1);
        std::array<long, N_dir> den_location {}; den_location.fill(-1);
        for ( int i_dir = 0; i_dir < N_dir; i_dir++ ) 
        {
            num_location[i_dir] = rcv_input.rcv_location[i_dir*2  ];
            den_location[i_dir] = rcv_input.rcv_location[i_dir*2+1];    
        }

        // global source index
        std::array<long, N_dir> index_rcv {}; index_rcv.fill(-1);

        // Convert the 'physical' location to global indices
        for ( int i_dir = 0; i_dir < N_dir; i_dir++ )
        {
            if ( grid_type[i_dir] == 'N' )
            {
                long num_rcv = num_location[i_dir] * ns_input::den_dx_soln;
                long den_rcv = den_location[i_dir] * ns_input::num_dx_soln;
                if ( den_rcv == 0 ) { printf("den_rcv should not be zero.\n"); fflush(stdout); exit(0); }
                if ( num_rcv % den_rcv != 0 )
                    { printf("Cannot convert physical location to integer index on direction %d ---- abort.\n", i_dir); fflush(stdout); exit(0); }
                index_rcv[i_dir] = num_rcv / den_rcv;
            }
            else if ( grid_type[i_dir] == 'M' )
            {
                long num_rcv = 2 * num_location[i_dir] * ns_input::den_dx_soln - den_location[i_dir] * ns_input::num_dx_soln;
                long den_rcv = 2 * den_location[i_dir] * ns_input::num_dx_soln;
                if ( den_rcv == 0 ) { printf("den_rcv should not be zero.\n"); fflush(stdout); exit(0); }
                if ( num_rcv % den_rcv != 0 )
                    { printf("Cannot convert physical location to integer index on direction %d ---- abort.\n", i_dir); fflush(stdout); exit(0); }
                index_rcv[i_dir] = num_rcv / den_rcv;
            }
            else
                { printf("Grid type unrecognized %c.\n", grid_type[i_dir]); fflush(stdout); exit(0); }
        }

        // Restore the if statement
        if ( grid_rcv->G_ix_bgn <= index_rcv[0] && index_rcv[0] <= grid_rcv->G_ix_end && 
             grid_rcv->G_iy_bgn <= index_rcv[1] && index_rcv[1] <= grid_rcv->G_iy_end )
        {
            // increment the number of receivers for src_index on grid_type
            Map_grid_N_rcvs.at(grid_type) += 1;

            struct_rcv_forward rcv_forward {};
            rcv_forward.rcv_index = rcv_input.rcv_index;
            rcv_forward.ix_rcv = index_rcv[0] - grid_rcv->G_ix_bgn;
            rcv_forward.iy_rcv = index_rcv[1] - grid_rcv->G_iy_bgn;
            rcv_forward.I_RCV  = rcv_forward.ix_rcv * grid_rcv->G_size_y + rcv_forward.iy_rcv;

            Map_grid_struct_rcv_forward.at(grid_type).push_back( rcv_forward );

            printf( " ---- Found receiver at (%3d %3d - %9d) on grid (%c %c) of sizes (%3d %3d)\n", 
                    rcv_forward.ix_rcv, rcv_forward.iy_rcv, rcv_forward.I_RCV, 
                    grid_rcv->G_type_x,  grid_rcv->G_type_y, 
                    grid_rcv->G_size_x,  grid_rcv->G_size_y );
            fflush(stdout);

            Map_grid_record_rcv.at(grid_type).push_back( run_time_matrix<ns_type::host_precision> {} );
            Map_grid_RESULT_rcv.at(grid_type).push_back( run_time_matrix<ns_type::host_precision> {} );
            Map_grid_misfit_rcv.at(grid_type).push_back( 0. );
        }
    }

    for ( const auto & iter_map : Map_Grid_pointers ) 
    {
        std::array<char, N_dir> grid_type = iter_map.first;
        Class_Grid * grid_rcv = iter_map.second;

        if ( static_cast<unsigned long> ( Map_grid_N_rcvs.at(grid_type) ) != Map_grid_struct_rcv_forward.at(grid_type).size()
          || static_cast<unsigned long> ( Map_grid_N_rcvs.at(grid_type) ) != Map_grid_record_rcv.at(grid_type).size() 
          || static_cast<unsigned long> ( Map_grid_N_rcvs.at(grid_type) ) != Map_grid_RESULT_rcv.at(grid_type).size()
          || static_cast<unsigned long> ( Map_grid_N_rcvs.at(grid_type) ) != Map_grid_misfit_rcv.at(grid_type).size() )
        { printf("Number of collected receivers do not match.\n"); fflush(stdout); exit(0); }

        if ( Map_grid_N_rcvs.at(grid_type) != 0 )
        {
            printf( " ---- Collected %3d receivers on grid (%c %c)\n", Map_grid_N_rcvs.at(grid_type), grid_type.at(0), grid_type.at(1) ); fflush(stdout);
        }

        // Allocate memory to store the solution at the receivers
        for ( auto& record_rcv : Map_grid_record_rcv.at(grid_type) ) 
            { record_rcv.allocate_memory( grid_rcv->N_soln , Nt ); }
        for ( auto& RESULT_rcv : Map_grid_RESULT_rcv.at(grid_type) ) 
            { RESULT_rcv.allocate_memory( grid_rcv->N_soln , Nt ); }
    }
} // Class_Forward_Specs::process_rcv_locations ()



void Class_Forward_Specs::output_solution_record ( std::string folder_name )
{
    for ( const auto & iter_map : Map_grid_N_rcvs ) // go through the possible grid types
    {
        auto & grid_type = iter_map.first;
        auto & N_rcvs    = iter_map.second;
    
        if ( N_rcvs > 0 )      // if number of receiver on this grid is larger than 0  
        {
            Class_Grid * grid_rcv = Map_Grid_pointers.at(grid_type);
            auto & Vec_struct_rcv = Map_grid_struct_rcv_forward.at(grid_type);
            auto & Vec_RESULT_rcv = Map_grid_RESULT_rcv.at(grid_type);

            for ( int i_rcv = 0; i_rcv < N_rcvs; i_rcv++ )  // loop through the receivers
            {                    
                auto & solution_record = Vec_RESULT_rcv.at(i_rcv);
                if ( solution_record.rows != grid_rcv->N_soln || solution_record.cols != Nt ) 
                    { printf("Recorded data size error.\n"); fflush(stdout); exit(0); }


                std::array<char, ns_forward::N_dir> physical_grid_type;
                physical_grid_type.at(0) = grid_type.at( ns_forward::XY_to_01 ('X') );
                physical_grid_type.at(1) = grid_type.at( ns_forward::XY_to_01 ('Y') );

                std::string file_name = folder_name + "/Signal"
                                      + "_" + "src" 
                                      + "_" + std::to_string( src_forward.src_index )                                      // src_index
                                      + "_" + src_forward.grid_type.at(1) + src_forward.grid_type.at(0)                    // src grid type
                                      + "_" + std::to_string( src_forward.iy_src )                                         // src location
                                      + "_" + std::to_string( src_forward.ix_src )                                         // src location
                                      + "_" + "rcv"
                                      + "_" + std::to_string( Vec_struct_rcv.at(i_rcv).rcv_index )                         // rcv_index
                                      + "_" + grid_type.at(1) + grid_type.at(0)                                            // rcv grid type
                                      + "_" + std::to_string( Vec_struct_rcv.at(i_rcv).iy_rcv )                            // rcv location
                                      + "_" + std::to_string( Vec_struct_rcv.at(i_rcv).ix_rcv );                           // rcv location

                if ( bool_visual )
                {
                    FILE * fp = fopen( ( file_name + ".txt" ).c_str(), "w" );
                    for ( int it = 0; it < Nt; it++ ) 
                    {
                        fprintf( fp, "% 6d  ", it ); 
                        for ( int i_field = 0; i_field < grid_rcv->N_soln; i_field++ )               // loop through the fields on this grid
                            { fprintf( fp, "% 16.15e  ", ns_input::source_scaling * (double) solution_record.at(i_field,it) ); }
                        fprintf( fp, "\n"); 
                    }
                    fclose(fp);
                }

            }
        }
    }
} // output_solution_record ()



void Class_Forward_Specs::output_energy ( std::string folder_name )
{
    std::string file_name = folder_name + "/Energy" 
                          + "_" + "src" 
                          + "_" + std::to_string( src_forward.src_index )                                      // src_index
                          + "_" + src_forward.grid_type.at(1) + src_forward.grid_type.at(0)                    // src grid type
                          + "_" + std::to_string( src_forward.iy_src )                                         // src location
                          + "_" + std::to_string( src_forward.ix_src )                                         // src location
                          + ".txt";

    FILE * fp = fopen( file_name.c_str() , "w" );

    for ( int it = 0; it < Nt; it++ ) 
    { 
        fprintf( fp, "%16.15e\n", ns_input::source_scaling * ns_input::source_scaling * 
                                  ( ns_input::Record_E_k .at(it) + 
                                  ( ns_input::Record_E_p0.at(it) + 
                                    ns_input::Record_E_p1.at(it) ) / 2. ) ); 
    }
    fclose(fp);

    std::vector<double> E (Nt);
    for ( int it = 0; it < Nt; it++ )
    {
        E.at(it) = ns_input::source_scaling * ns_input::source_scaling * 
                   ( ns_input::Record_E_k .at(it) + 
                   ( ns_input::Record_E_p0.at(it) + 
                     ns_input::Record_E_p1.at(it) ) / 2. );
    }

    auto max_iter = std::max_element( E.begin() + Nt - Nt / 2, E.end() );
    auto min_iter = std::min_element( E.begin() + Nt - Nt / 2, E.end() );

    printf( "Last Nt/2 - max: % 12.11e min: % 12.11e diff (relative): % 12.11e\n" , 
            (double) *max_iter , (double) *min_iter , (double) ( (*max_iter - *min_iter) / *max_iter ) );
} // output_energy ()



void Class_Forward_Specs::calculate_data_misfit ()
{
    this->data_misfit = 0.;
    for ( const auto & iter_map : Map_grid_N_rcvs )    // go through the possible grid types
    {
        auto & grid_type = iter_map.first;
        auto & N_rcvs    = iter_map.second;

        if ( N_rcvs > 0 )                              // if number of receiver on this grid is larger than 0
        {
            const int & N_soln = Map_Grid_pointers.at(grid_type)->N_soln;

            auto & Vec_record_rcv = Map_grid_record_rcv.at(grid_type);
            auto & Vec_RESULT_rcv = Map_grid_RESULT_rcv.at(grid_type);

            for ( int i_rcv = 0; i_rcv < N_rcvs; i_rcv++ )    // loop through the receivers
            {                    
                auto & solution_record = Vec_record_rcv.at(i_rcv);
                auto & solution_RESULT = Vec_RESULT_rcv.at(i_rcv);

                double data_misfit_rcv = 0.;
                for ( int it = 0; it < Nt; it++ )
                    for ( int i_field = 0; i_field < N_soln; i_field++ )
                        { data_misfit_rcv += std::pow( (double) ( solution_RESULT.at(i_field,it) 
                                                                - solution_record.at(i_field,it) ) , 2 ); }

                Map_grid_misfit_rcv.at(grid_type).at(i_rcv) = data_misfit_rcv;
                this->data_misfit                          += data_misfit_rcv;
            }
        }
    }
} // calculate_data_misfit ()