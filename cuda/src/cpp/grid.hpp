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

        // the number of (wave)fields associated with this grid
        int N_soln = -1;  // 2 for NN ; 0 for MM ; 1 for the others
        // the number of parameters associated with this grid
        int N_prmt = -1;  // 2 for NN ; 0 for MM ; 1 for the others
        // the number of parameters associated with this grid
        int N_enrg = -1;  // 2 for NN ; 0 for MM ; 1 for the others

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
        // #include "grid_update.tpp"

        void adjust_parameters_energy_periodic ();
        
        double energy_calculation ();

        template<typename T>
        void interpolate_forward_parameter (  
                                             run_time_vector< T > & forward_P );

        void retrieve_forward_parameter ( Class_Inverse_Specs &Inv_Specs );

};

#endif

