#ifndef GRID_H
#define GRID_H

#include <math.h>
#include <vector>
#include <map>
#include <array>
#include <string>
#include <assert.h>

#include "container_run_time.hpp"
#include "namespace_forward.hpp"
#include "namespace_input.hpp"

class Class_Grid 
{
    public:
        char G_type_x;
        char G_type_y;
        int  G_size_x;
        int  G_size_y;
        int stride_x;
        int stride_y;
        int N_modulo_x = 1<<30;
        int N_modulo_y = 1<<30;

        int SL_external = 1<<30;
        int SL_internal = 1<<30;

        run_time_vector<ns_type::host_precision> stencil_dt_dx;

        int N_soln = -1;
        int N_enrg = -1;

        std::string grid_name = "U"; 
        bool free_surface_update = false;

        Class_Grid() = default;

        void set_grid_parameters ( std::array<char,ns_forward::N_dir> GT , bool bool_energy ) 
        {
            if ( GT.at(0) == 'N' || GT.at(0) == 'M' ) { G_type_x = GT.at(0); } else { printf("Grid type can only be N or M.\n"); exit(0); }
            if ( GT.at(1) == 'N' || GT.at(1) == 'M' ) { G_type_y = GT.at(1); } else { printf("Grid type can only be N or M.\n"); exit(0); }

            if ( G_type_x == 'N' && G_type_y == 'N' ) { this->grid_name = "Sxy"; }
            if ( G_type_x == 'M' && G_type_y == 'M' ) { this->grid_name = "SMM"; }
            if ( G_type_x == 'M' && G_type_y == 'N' ) { this->grid_name = "Vx";  }
            if ( G_type_x == 'N' && G_type_y == 'M' ) { this->grid_name = "Vy";  }

            if ( strcmp( grid_name.c_str(), "Vx" ) == 0 || strcmp( grid_name.c_str(), "Vy" ) == 0 ) { this->free_surface_update = true; }

            if ( G_type_x == 'N' ) { G_size_x = ns_input::Nx_soln; }  if ( G_type_x == 'M' ) { G_size_x = ns_input::Mx_soln; }
            if ( G_type_y == 'N' ) { G_size_y = ns_input::Ny_soln; }  if ( G_type_y == 'M' ) { G_size_y = ns_input::My_soln; }
            
            this->N_modulo_x = this->G_size_x;  if ( G_type_x == 'N' ) { this->N_modulo_x -= 1; }
            this->N_modulo_y = this->G_size_y;  if ( G_type_y == 'N' ) { this->N_modulo_y -= 1; }

            this->N_soln = 1;  if ( strcmp( grid_name.c_str(), "SMM" ) == 0 ) { this->N_soln = 2; }
            this->N_enrg = 1;  if ( strcmp( grid_name.c_str(), "SMM" ) == 0 ) { this->N_enrg = 2; }
            if ( bool_energy == false ) { this->N_enrg = 0; }
        }

        void set_forward_operators ()
        {
            using ns_input::dt;
            using ns_input::dx;

            this->stride_x = this->G_size_y;
            this->stride_y =              1;

            // Removed Map_stencil_shift allocation

            // Dummy implementation to keep it compiling without full operators
            // if they are needed by cuda_Class_Grid_initialize.
            // You may need to restore full implementation if these maps need to be non-empty.
        }
};

#endif