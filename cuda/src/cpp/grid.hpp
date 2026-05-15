#ifndef GRID_H
#define GRID_H

#include <math.h>
#include <vector>

#include "container_run_time.hpp"
#include "namespace_forward.hpp"
#include "namespace_input.hpp"

class Class_Inverse_Specs;

class Class_Grid 
{
    public:

        // pointers to the grids that that this grid interacts with
        Class_Grid * pntr_Grid_x;
        Class_Grid * pntr_Grid_y;

        // type of this grid (i.e., is it an N grid or M grid, direction-wise)
        char G_type_x;
        char G_type_y;

        // size of this grid (in terms of number of grid points)
        int  G_size_x;
        int  G_size_y;

        // stride: change in the 2D vector index caused by change in grid index in x or y direction by unit 1
        int stride_x;
        int stride_y;

        // beginning and ending indices (direction-wise)
        int G_ix_bgn, G_ix_end;     // 0 , G_size_x
        int G_iy_bgn, G_iy_end;     // 0 , G_size_y

        int N_modulo_x = 1<<30;  // Used to map the indices for periodic BC; needed when using OR not using the % operator
        int N_modulo_y = 1<<30;  // Used to map the indices for periodic BC; needed when using OR not using the % operator

        int SL_external = 1<<30;  // SL : stencil length
        int SL_internal = 1<<30;  // SL : stencil length

        // the number of (wave)fields associated with this grid
        int N_soln = -1;  // 2 for NN ; 0 for MM ; 1 for the others
        // the number of parameters associated with this grid
        int N_prmt = -1;  // 2 for NN ; 0 for MM ; 1 for the others
        // the number of parameters associated with this grid
        int N_enrg = -1;  // 2 for NN ; 0 for MM ; 1 for the others

        // solution variables on this grid (they have the same size as this grid)
        std::vector< run_time_vector<ns_type::host_precision> > Vec_soln;

        // right hand side variables on this grid (they have the same size as this grid)
        std::vector< run_time_vector<ns_type::host_precision> > Vec_rths;

        // compensation variables on this grid (they have the same size as this grid)
        std::vector< run_time_vector<ns_type::host_precision> > Vec_cpst;

        // grid name (used in src_rcv.cpp to assign to c_source_type and in output_processing.hpp to define output file name)
        std::string grid_name = "U"; 

        // physical parameters on this grid (they have the same size as this grid)
        std::vector< run_time_vector<ns_type::host_precision> > Vec_prmt;

        // extra physical parameters (needed for energy calculation) on this grid (they have the same size as this grid)
        std::vector< run_time_vector< double > > Vec_prmt_enrg;

        bool free_surface_update = false;    // set to true for Vx, Vy, Vz ; false for all stresses

        // ------ the stencil from ns_forward (with unit length) divided by dx used in this simulation
        run_time_vector<ns_type::host_precision> stencil_dt_dx {4};

        ns_input::InputParams &params;

        // constructor
        Class_Grid ( ns_input::InputParams &p ) : params(p) { }        

        
        //-----------------------------------------------//
        //------------- Function defintiion -------------//
        //-----------------------------------------------//
        void set_grid_parameters ( std::array<char,ns_forward::N_dir> GT , bool bool_energy ) 
        {
            // Grid type
            if ( GT.at(0) == 'N' || GT.at(0) == 'M' ) { G_type_x = GT.at(0); } else { printf("Grid type can only be N or M.\n"); fflush(stdout); exit(0); }
            if ( GT.at(1) == 'N' || GT.at(1) == 'M' ) { G_type_y = GT.at(1); } else { printf("Grid type can only be N or M.\n"); fflush(stdout); exit(0); }

            if ( G_type_x == 'N' && G_type_y == 'N' ) { this->grid_name = "Sxy"; }
            if ( G_type_x == 'M' && G_type_y == 'M' ) { this->grid_name = "SMM"; }

            if ( G_type_x == 'M' && G_type_y == 'N' ) { this->grid_name = "Vx";  }
            if ( G_type_x == 'N' && G_type_y == 'M' ) { this->grid_name = "Vy";  }

            // If this grid is involved in boundary treatment related to free surface
            if ( strcmp( grid_name.c_str(), "Vx" ) == 0 || strcmp( grid_name.c_str(), "Vy" ) == 0 ) { this->free_surface_update = true; }

            // Grid size 
            if ( G_type_x == 'N' ) { G_size_x = params.Nx_soln; }  if ( G_type_x == 'M' ) { G_size_x = params.Mx_soln; }
            if ( G_type_y == 'N' ) { G_size_y = params.Ny_soln; }  if ( G_type_y == 'M' ) { G_size_y = params.My_soln; }
            
            G_ix_bgn = 0;  G_ix_end = G_size_x;
            G_iy_bgn = 0;  G_iy_end = G_size_y;

            this->N_modulo_x = this->G_size_x;  if ( G_type_x == 'N' ) { this->N_modulo_x -= 1; }
            this->N_modulo_y = this->G_size_y;  if ( G_type_y == 'N' ) { this->N_modulo_y -= 1; }

            // Number of solution variables residing on this grid 
            this->N_soln = 1;  if ( strcmp( grid_name.c_str(), "SMM" ) == 0 ) { this->N_soln = 2; }
            this->N_prmt = 1;  if ( strcmp( grid_name.c_str(), "SMM" ) == 0 ) { this->N_prmt = 2; }
            this->N_enrg = 1;  if ( strcmp( grid_name.c_str(), "SMM" ) == 0 ) { this->N_enrg = 2; }
            if ( bool_energy == false ) { this->N_enrg = 0; }

            // allocate space for solution fields
            if ( this->N_soln > 0 ) { this->Vec_soln     .reserve(this->N_soln); }
            if ( this->N_prmt > 0 ) { this->Vec_prmt     .reserve(this->N_prmt); }

            if ( this->N_soln > 0 ) { this->Vec_rths     .reserve(this->N_soln); }
            if ( this->N_soln > 0 ) { this->Vec_cpst     .reserve(this->N_soln); }

            if ( this->N_enrg > 0 ) { this->Vec_prmt_enrg.reserve(this->N_enrg); }

            for ( int i = 0; i < this->N_soln; i++ ) { this->Vec_soln     .push_back( run_time_vector<ns_type::host_precision> { G_size_x * G_size_y } ); }
            for ( int i = 0; i < this->N_prmt; i++ ) { this->Vec_prmt     .push_back( run_time_vector<ns_type::host_precision> { G_size_x * G_size_y } ); }
            
            for ( int i = 0; i < this->N_soln; i++ ) { this->Vec_rths     .push_back( run_time_vector<ns_type::host_precision> { G_size_x * G_size_y } ); }
            for ( int i = 0; i < this->N_soln; i++ ) { this->Vec_cpst     .push_back( run_time_vector<ns_type::host_precision> { G_size_x * G_size_y } ); }

            for ( int i = 0; i < this->N_enrg; i++ ) { this->Vec_prmt_enrg.push_back( run_time_vector< double >                { G_size_x * G_size_y } ); }

        } // set_grid_parameters()


        void set_grid_pointers ( std::map< std::array<char, ns_forward::N_dir> , Class_Grid * > & Map_Grid_pointers )
        {
            // Empty in minimized state
        } // set_grid_pointers()


        void set_forward_operators ()
        {
            double dt = params.dt;
            double dx = params.dx;

            assert ( this->stencil_dt_dx.length == ns_forward::stencil.length );

            for ( int i=0; i<stencil_dt_dx.length; i++ ) 
                { this->stencil_dt_dx.at(i) = ns_forward::stencil.at(i) * (dt / dx); }

            SL_external = 1<<30;
            if ( this->G_type_y == 'M' ) { SL_external = 1; };
            if ( this->G_type_y == 'N' ) { SL_external = 2; };

            SL_internal = ns_forward::length_stencil - SL_external;

            this->stride_x = this->G_size_y;
            this->stride_y =              1;

        } // set_forward_operators()


        // #include "grid_drvt.tpp"

        #include "grid_update.tpp"

        template<typename T>
        void fast2sum( T const a , T const b ,
                       T &     s , T &     t )
        {
              s = a + b;
            T z = s - a;
              t = b - z;
        }

        template<typename T>
        void slow2sum( T const a , T const b ,
                       T &     s , T &     t )
        {
                s =   a + b;
            T p_a =   s - b;
            T p_b =   s - p_a;

            T d_a =   a - p_a;
            T d_b =   b - p_b;

                t = d_a + d_b;
        } 

        template<int cpst_N = 0 , char cpst_S = '0'>
        void update_cpst ()
        {
                 if ( cpst_N ==  0 && cpst_S == '0' ) { update    <ns_type::host_precision> (); }

            else if ( cpst_N == -3 && cpst_S == 'C' ) { update_3C <ns_type::host_precision> (); }
            else if ( cpst_N == -3 && cpst_S == 'R' ) { update_3R <ns_type::host_precision> (); }

            else if ( cpst_N ==  3 && cpst_S == 'C' ) { update_3C_fast2sum <ns_type::host_precision> (); }
            else if ( cpst_N ==  3 && cpst_S == 'R' ) { update_3R_fast2sum <ns_type::host_precision> (); }

            else if ( cpst_N ==  6 && cpst_S == 'C' ) { update_6C_slow2sum <ns_type::host_precision> (); }
            else if ( cpst_N ==  6 && cpst_S == 'R' ) { update_6R_slow2sum <ns_type::host_precision> (); }

            else 
                { printf("Unrecognized update function %s %d\n", __FILE__, __LINE__); exit(0); }
        }


        void define_parameters_energy ();

        void adjust_parameters_energy_periodic ();
        
        double energy_calculation ();

        template<typename T>
        void interpolate_forward_parameter (  
                                             run_time_vector< T > & forward_P );

        void retrieve_forward_parameter ( Class_Inverse_Specs &Inv_Specs );

};

#endif
