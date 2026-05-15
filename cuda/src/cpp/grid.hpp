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
    // NOTE: The suffixes "_x" and "_y" below refer to the "data" direction; 
    //       they are synonymous with 0 and 1, i.e, the slow and fast direction
    //       that are used to determine data layout. 
    //           The "physical" direction used for specifying and processing
    //       input data, such as in InputFile.txt, SrcRcv.txt, inverse.cpp, and
    //       input_processing.cpp, are mapped to 0 or 1 using a function defined
    //       in namespace_forward.hpp. 
    //           Specifically, 'X' ('Y') does not have to correspond to 'x' ('y').
    //
    //       This note is general and not limited to this file.

    // NOTE on NOTE: This XY and xy distinction start to smell like a footgun 
    //               within a week. 

    public:

        // pointers to the grids that that this grid interacts with
        Class_Grid * pntr_Grid_x;
        Class_Grid * pntr_Grid_y;
        std::map< char , Class_Grid * > Map_pntr_grid;    // char : c_dir

        // type of this grid (i.e., is it an N grid or M grid, direction-wise)
        char G_type_x;
        char G_type_y;
        std::map< char , char > Map_G_type;    // key : c_dir ; value : g_type 

        // size of this grid (in terms of number of grid points)
        int  G_size_x;
        int  G_size_y;
        std::map< char , int  > Map_G_size;    // key : c_dir

        // stride: change in the 2D vector index caused by change in grid index in x or y direction by unit 1
        int stride_x;
        int stride_y;
        std::map< char , int > Map_stride;     // key : c_dir
        // NOTE: If the interacting direction is 'dir', the strides AT and AFTER (in the order of x , y) 'dir' 
        //       do not differ between 'this' grid and the 'pntr' grid (i.e., the interacting grid). This is a 
        //       consequence following the observation that only G_size_'dir' differs between the two grids, and
        //       how stride_'dir' is defined.
        //           stride_x = G_size_y;
        //           stride_y =        1;
        //       AT or AFTER 'dir', the difference in G_size_'dir' no longer make a difference since it does not
        //       enter the definition of stride_'dir'.


        // beginning and ending indices (direction-wise)
        int G_ix_bgn, G_ix_end;     // 0 , G_size_x
        int G_iy_bgn, G_iy_end;     // 0 , G_size_y
        // NOTE: Since we don't consider MPI for forward, the above variables have trivial values as shown in the comment;
        //       They are kept for API stability.


        int N_modulo_x = 1<<30;  // Used to map the indices for periodic BC; needed when using OR not using the % operator
        int N_modulo_y = 1<<30;  // Used to map the indices for periodic BC; needed when using OR not using the % operator
        std::map< char , int > Map_N_modulo;    // key : c_dir

        std::map< char , int > Map_interior_BGN;   // WARNING: should only be used for the direction along which the derivative is taken
        std::map< char , int > Map_interior_END;   // WARNING: should only be used for the direction along which the derivative is taken

        int SL_external = 1<<30;  // SL : stencil length
        int SL_internal = 1<<30;  // SL : stencil length


        // derivatives on this grid (they have the same size as this grid)
        std::map< char, run_time_vector<ns_type::host_precision> > Map_v_D;    // key : c_dir

        // ---- the number of (wave)fields associated with this grid
        int N_soln = -1;  // 2 for NN ; 0 for MM ; 1 for the others
        // ---- the number of parameters associated with this grid
        int N_prmt = -1;  // 2 for NN ; 0 for MM ; 1 for the others
        // ---- the number of parameters associated with this grid
        int N_enrg = -1;  // 2 for NN ; 0 for MM ; 1 for the others

        // solution variables on this grid (they have the same size as this grid)
        std::vector< run_time_vector<ns_type::host_precision> > Vec_soln;

        // right hand side variables on this grid (they have the same size as this grid)
        std::vector< run_time_vector<ns_type::host_precision> > Vec_rths;
        // [2023/06/17] NOTE: The right hand side variables are R in dU / dt = R.

        // compensation variables on this grid (they have the same size as this grid)
        std::vector< run_time_vector<ns_type::host_precision> > Vec_cpst;

        // grid name (used in src_rcv.cpp to assign to c_source_type and in output_processing.hpp to define output file name)
        std::string grid_name = "U"; 


        std::map< char , std::map< char , run_time_vector<ns_type::host_precision> > > Map_buffer_SEND;  // first char : {'x','y'}; second char : {'L','R'}
        std::map< char , std::map< char , run_time_vector<ns_type::host_precision> > > Map_buffer_RECV;  // first char : {'x','y'}; second char : {'L','R'}


        // pointers to solution variables on the interacting grids
        std::map< char , run_time_vector<ns_type::host_precision> * > Map_pntr_soln;

        // pointers to solution variables on the interacting grids
        std::map< char , run_time_vector<ns_type::host_precision> * > Map_pntr_rths;

        // physical parameters on this grid (they have the same size as this grid)
        std::vector< run_time_vector<ns_type::host_precision> > Vec_prmt;

        // extra physical parameters (needed for energy calculation) on this grid (they have the same size as this grid)
        std::vector< run_time_vector< double > > Vec_prmt_enrg;


        bool free_surface_update = false;    // set to true for Vx, Vy, Vz ; false for all stresses


        // ------ the stencil from ns_forward (with unit length) divided by dx used in this simulation
        run_time_vector<ns_type::host_precision> stencil_dt_dx {4};

        // boundary operators from ns_forward (with unit length) divided by dx used in this simulation
        std::map< std::pair<char,char> , run_time_matrix<ns_type::host_precision> > Map_D_bdry;
        

        std::map< char , run_time_vector<int> > Map_stencil_shift;    // index difference when applying stencil to update the solution        
        

        // 1D operators related to projection and its inverse operation (weighted by the diagonal components of the norm matrices) 
        // to weakly enforce the boundary conditions
        std::map< std::pair<char,char> , run_time_vector<ns_type::host_precision> > Map_projection;          // key : { c_dir , c_LR }
        std::map< std::pair<char,char> , run_time_vector<ns_type::host_precision> > Map_A_bdry_diag;         // key : { c_dir , c_LR }
        std::map< std::pair<char,char> , run_time_vector<ns_type::host_precision> > Map_A_inv_projection;    // key : { c_dir , c_LR }


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

            this->Map_G_type['x'] = G_type_x;
            this->Map_G_type['y'] = G_type_y;

            if ( G_type_x == 'N' && G_type_y == 'N' ) { this->grid_name = "Sxy"; }
            if ( G_type_x == 'M' && G_type_y == 'M' ) { this->grid_name = "SMM"; }

            if ( G_type_x == 'M' && G_type_y == 'N' ) { this->grid_name = "Vx";  }  // Pay attention here; we are using small 'x'
            if ( G_type_x == 'N' && G_type_y == 'M' ) { this->grid_name = "Vy";  }  // Pay attention here; we are using small 'y'

            // Should remove x and y from the above naming, use N and M only, leave the mapping to physics to some other file higher in the chain


            // If this grid is involved in boundary treatment related to free surface
            if ( strcmp( grid_name.c_str(), "Vx" ) == 0 || strcmp( grid_name.c_str(), "Vy" ) == 0 ) { this->free_surface_update = true; }

            // Grid size 
            if ( G_type_x == 'N' ) { G_size_x = params.Nx_soln; }  if ( G_type_x == 'M' ) { G_size_x = params.Mx_soln; }
            if ( G_type_y == 'N' ) { G_size_y = params.Ny_soln; }  if ( G_type_y == 'M' ) { G_size_y = params.My_soln; }
            
            this->Map_G_size['x'] = G_size_x;
            this->Map_G_size['y'] = G_size_y;

            G_ix_bgn = 0;  G_ix_end = G_size_x;
            G_iy_bgn = 0;  G_iy_end = G_size_y;

            this->N_modulo_x = this->G_size_x;  if ( G_type_x == 'N' ) { this->N_modulo_x -= 1; }
            this->N_modulo_y = this->G_size_y;  if ( G_type_y == 'N' ) { this->N_modulo_y -= 1; }

            this->Map_N_modulo['x'] = this->N_modulo_x;
            this->Map_N_modulo['y'] = this->N_modulo_y;


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

            // allocate space for derivatives associated with this Grid
            // ---- in 2D, all grids need both x- and y- derivative
            this->Map_v_D['x'].allocate_memory( G_size_x * G_size_y );
            this->Map_v_D['y'].allocate_memory( G_size_x * G_size_y );


            for ( const char& c_dir : {'x','y'} )
            {
                long buffer_size = -(1<<30);
                if ( c_dir == 'x' ) { buffer_size = this->G_size_y; }
                if ( c_dir == 'y' ) { buffer_size = this->G_size_x; }

                for ( const char& c_LR : {'L','R'} )
                {
                    Map_buffer_SEND[c_dir][c_LR].allocate_memory(buffer_size);
                    Map_buffer_RECV[c_dir][c_LR].allocate_memory(buffer_size);
                }
            }

        } // set_grid_parameters()


        //-----------------------------------------------//
        //------------- Function defintiion -------------//
        //-----------------------------------------------//
        void set_grid_pointers ( std::map< std::array<char, ns_forward::N_dir> , Class_Grid * > & Map_Grid_pointers )
        {
            char P_type_x = ( G_type_x == 'N' ? 'M' : 'N' );
            char P_type_y = ( G_type_y == 'N' ? 'M' : 'N' );

            this->pntr_Grid_x = Map_Grid_pointers.at( { P_type_x , G_type_y } ) ; 
            this->pntr_Grid_y = Map_Grid_pointers.at( { G_type_x , P_type_y } ) ; 

            Map_pntr_grid['x'] = pntr_Grid_x;
            Map_pntr_grid['y'] = pntr_Grid_y;


            // check the sizes
            if ( G_type_x == 'N' ) assert ( G_size_x - pntr_Grid_x->G_size_x ==  1 );
            if ( G_type_x == 'M' ) assert ( G_size_x - pntr_Grid_x->G_size_x == -1 );

            if ( G_type_y == 'N' ) assert ( G_size_y - pntr_Grid_y->G_size_y ==  1 );
            if ( G_type_y == 'M' ) assert ( G_size_y - pntr_Grid_y->G_size_y == -1 );


            // assign to pointers to the solution variables on the interacting grids
            if ( pntr_Grid_x->N_soln == 2 ) { Map_pntr_soln['x'] = & pntr_Grid_x->Vec_soln.at(0); }
            if ( pntr_Grid_y->N_soln == 2 ) { Map_pntr_soln['y'] = & pntr_Grid_y->Vec_soln.at(1); }

            if ( pntr_Grid_x->N_soln == 1 ) { Map_pntr_soln['x'] = & pntr_Grid_x->Vec_soln.at(0); }
            if ( pntr_Grid_y->N_soln == 1 ) { Map_pntr_soln['y'] = & pntr_Grid_y->Vec_soln.at(0); }


            // assign to pointers to the right hand side variables on the interacting grids
            if ( pntr_Grid_x->N_soln == 2 ) { Map_pntr_rths['x'] = & pntr_Grid_x->Vec_rths.at(0); }
            if ( pntr_Grid_y->N_soln == 2 ) { Map_pntr_rths['y'] = & pntr_Grid_y->Vec_rths.at(1); }

            if ( pntr_Grid_x->N_soln == 1 ) { Map_pntr_rths['x'] = & pntr_Grid_x->Vec_rths.at(0); }
            if ( pntr_Grid_y->N_soln == 1 ) { Map_pntr_rths['y'] = & pntr_Grid_y->Vec_rths.at(0); }

        } // set_grid_pointers()


        //-----------------------------------------------//
        //------------- Function defintiion -------------//
        //-----------------------------------------------//
        void set_forward_operators ()
        {
            double dt = params.dt;
            double dx = params.dx;

            assert ( this->stencil_dt_dx.length == ns_forward::stencil.length );

            for ( int i=0; i<stencil_dt_dx.length; i++ ) 
                { this->stencil_dt_dx.at(i) = ns_forward::stencil.at(i) * (dt / dx); }

            // NOTE: We assume that x boundaries are weak; y is strong.
            for ( const char& c_dir : {'x'} )
            {
                char c_type = this->Map_G_type.at(c_dir);

                if ( c_type == 'N' ) 
                { 
                    constexpr auto & D_L = ns_forward::W_intr.DM_L;
                    constexpr auto & D_R = ns_forward::W_intr.DM_R;

                    this->Map_D_bdry[{c_dir,'L'}].allocate_memory( D_L.rows , D_L.cols );
                    this->Map_D_bdry[{c_dir,'R'}].allocate_memory( D_R.rows , D_R.cols );

                    for ( int i=0; i<D_L.rows; i++ )
                    for ( int j=0; j<D_L.cols; j++ )
                        { this->Map_D_bdry.at({c_dir,'L'}).at(i,j) = D_L.at(i,j) * (dt / dx); }

                    for ( int i=0; i<D_R.rows; i++ )
                    for ( int j=0; j<D_R.cols; j++ )
                        { this->Map_D_bdry.at({c_dir,'R'}).at(i,j) = D_R.at(i,j) * (dt / dx); }
                }

                if ( c_type == 'M' ) 
                { 
                    constexpr auto & D_L = ns_forward::W_intr.DN_L;
                    constexpr auto & D_R = ns_forward::W_intr.DN_R;

                    this->Map_D_bdry[{c_dir,'L'}].allocate_memory( D_L.rows , D_L.cols );
                    this->Map_D_bdry[{c_dir,'R'}].allocate_memory( D_R.rows , D_R.cols );

                    for ( int i=0; i<D_L.rows; i++ )
                    for ( int j=0; j<D_L.cols; j++ )
                        { this->Map_D_bdry.at({c_dir,'L'}).at(i,j) = D_L.at(i,j) * (dt / dx); }

                    for ( int i=0; i<D_R.rows; i++ )
                    for ( int j=0; j<D_R.cols; j++ )
                        { this->Map_D_bdry.at({c_dir,'R'}).at(i,j) = D_R.at(i,j) * (dt / dx); }
                }
            }


            Map_interior_BGN['x'] =                  Map_D_bdry.at({'x','L'}).rows; 
            Map_interior_END['x'] = this->G_size_x - Map_D_bdry.at({'x','R'}).rows; 

            Map_interior_BGN['y'] =              0;
            Map_interior_END['y'] = this->G_size_y;

            if ( this->G_type_y == 'M' ) { Map_interior_BGN.at('y') += 1; Map_interior_END.at('y') -= 1; }
            if ( this->G_type_y == 'N' ) { Map_interior_BGN.at('y') += 2; Map_interior_END.at('y') -= 2; }


            SL_external = 1<<30;
            if ( this->G_type_y == 'M' ) { SL_external = 1; };
            if ( this->G_type_y == 'N' ) { SL_external = 2; };
            // NTOE: SL_external is the maximal number of pntr grid points needed (i.e., 
            //       those that need to be wrapped around) to fulfill the stencil, which 
            //       corresponds to the update of the left-most or right-most grid point 
            //       on this grid.

            SL_internal = ns_forward::length_stencil - SL_external;
            // NOTE: SL_internal is the internal part of the stencil for the left-most 
            //       or right-most grid point on this grid.


            this->stride_x = this->G_size_y;
            this->stride_y =              1;

            this->Map_stride['x'] = this->stride_x;
            this->Map_stride['y'] = this->stride_y;

            // for eye-checking the strides
            printf( "strides on grid (%c%c) : %d %d\n", G_type_x, G_type_y, this->Map_stride.at('x'), this->Map_stride.at('y') );

            for ( const char& c_dir : {'x','y'} )
            {
                this->Map_stencil_shift[c_dir].allocate_memory( this->stencil_dt_dx.length );

                if ( this->Map_G_type.at(c_dir) == 'N' ) { this->Map_stencil_shift.at(c_dir) = {-2, -1, 0, 1}; }
                if ( this->Map_G_type.at(c_dir) == 'M' ) { this->Map_stencil_shift.at(c_dir) = {-1,  0, 1, 2}; }

                // this->Map_stencil_shift.at(c_dir) *= this->Map_stride.at(c_dir);
            }


            // NOTE: Below, we prepare the projection and its norm-matrix-weighted transpose 
            //       needed for boundary treatment. 
            //           Again, we assume that all boundaries are weak.


            // NOTE: In secure_bdry_* (as well as calculate_derivative_*), derivative on THIS grid is calculated 
            //       so that solution on THIS grid can be updated. 
            //           This means that the solution that needed to be projected is from the 'interacting' grid,
            //       and that the projection operator should apply on solution from the 'interacting' grid.
            for ( const char& c_dir : {'x'} )
            // NOTE: We assume that x boundaries are weak; y is strong.
            {
                char c_type = this->Map_G_type.at(c_dir);

                if ( c_type == 'M' ) // ATTENTION to the c_type here.
                { 
                    constexpr auto & P_L = ns_forward::projection_operator.N_L;
                    constexpr auto & P_R = ns_forward::projection_operator.N_R;

                    this->Map_projection[{c_dir,'L'}].allocate_memory( P_L.length );
                    this->Map_projection[{c_dir,'R'}].allocate_memory( P_R.length );

                    for ( int i=0; i<P_L.length; i++ ) { this->Map_projection.at({c_dir,'L'}).at(i) = P_L.at(i); }
                    for ( int i=0; i<P_R.length; i++ ) { this->Map_projection.at({c_dir,'R'}).at(i) = P_R.at(i); }
                }

                if ( c_type == 'N' ) // ATTENTION to the c_type here.
                { 
                    constexpr auto & P_L = ns_forward::projection_operator.M_L;
                    constexpr auto & P_R = ns_forward::projection_operator.M_R;

                    this->Map_projection[{c_dir,'L'}].allocate_memory( P_L.length );
                    this->Map_projection[{c_dir,'R'}].allocate_memory( P_R.length );

                    for ( int i=0; i<P_L.length; i++ ) { this->Map_projection.at({c_dir,'L'}).at(i) = P_L.at(i); }
                    for ( int i=0; i<P_R.length; i++ ) { this->Map_projection.at({c_dir,'R'}).at(i) = P_R.at(i); }
                }
            }


            for ( const char& c_dir : {'x'} )
            // NOTE: We assume that x boundaries are weak; y is strong.
            {
                char c_type = this->Map_G_type.at(c_dir);

                if ( c_type == 'N' ) 
                { 
                    constexpr auto & A_L = ns_forward::A_diag.N_L;
                    constexpr auto & A_R = ns_forward::A_diag.N_R;

                    this->Map_A_bdry_diag[{c_dir,'L'}].allocate_memory( A_L.length );
                    this->Map_A_bdry_diag[{c_dir,'R'}].allocate_memory( A_R.length );

                    for ( int i=0; i<A_L.length; i++ ) { this->Map_A_bdry_diag.at({c_dir,'L'}).at(i) = A_L.at(i) * dx; }
                    for ( int i=0; i<A_R.length; i++ ) { this->Map_A_bdry_diag.at({c_dir,'R'}).at(i) = A_R.at(i) * dx; }

                    constexpr auto & P_L = ns_forward::projection_operator.N_L;
                    constexpr auto & P_R = ns_forward::projection_operator.N_R;

                    this->Map_A_inv_projection[{c_dir,'L'}].allocate_memory( P_L.length );
                    this->Map_A_inv_projection[{c_dir,'R'}].allocate_memory( P_R.length );

                    for ( int i=0; i<P_L.length; i++ ) 
                        { this->Map_A_inv_projection.at({c_dir,'L'}).at(i) = P_L.at(i) / ( dx * A_L.at(i) ); }
                    for ( int i=P_R.length; i>0; i-- )
                        { this->Map_A_inv_projection.at({c_dir,'R'}).at(P_R.length-i) = P_R.at(P_R.length-i) / ( dx * A_R.at(A_R.length-i) ); }

                    this->Map_A_inv_projection.at({c_dir,'L'}) *= dt;
                    this->Map_A_inv_projection.at({c_dir,'R'}) *= dt;
                }

                if ( c_type == 'M' ) 
                { 
                    constexpr auto & A_L = ns_forward::A_diag.M_L;
                    constexpr auto & A_R = ns_forward::A_diag.M_R;

                    this->Map_A_bdry_diag[{c_dir,'L'}].allocate_memory( A_L.length );
                    this->Map_A_bdry_diag[{c_dir,'R'}].allocate_memory( A_R.length );

                    for ( int i=0; i<A_L.length; i++ ) { this->Map_A_bdry_diag.at({c_dir,'L'}).at(i) = A_L.at(i) * dx; }
                    for ( int i=0; i<A_R.length; i++ ) { this->Map_A_bdry_diag.at({c_dir,'R'}).at(i) = A_R.at(i) * dx; }

                    constexpr auto & P_L = ns_forward::projection_operator.M_L;
                    constexpr auto & P_R = ns_forward::projection_operator.M_R;

                    this->Map_A_inv_projection[{c_dir,'L'}].allocate_memory( P_L.length );
                    this->Map_A_inv_projection[{c_dir,'R'}].allocate_memory( P_R.length );

                    for ( int i=0; i<P_L.length; i++ ) 
                        { this->Map_A_inv_projection.at({c_dir,'L'}).at(i) = P_L.at(i) / ( dx * A_L.at(i) ); }
                    for ( int i=P_R.length; i>0; i-- )
                        { this->Map_A_inv_projection.at({c_dir,'R'}).at(P_R.length-i) = P_R.at(P_R.length-i) / ( dx * A_R.at(A_R.length-i) ); }

                    this->Map_A_inv_projection.at({c_dir,'L'}) *= dt;
                    this->Map_A_inv_projection.at({c_dir,'R'}) *= dt;                    
                }
            }

            // NOTE: the following comments are for eye-checking the projection and A_inv_projection operators
            /*
            for ( const char& c_dir : {'x','y'} )
            {
                for ( const char& c_LR : {'L','R'} )
                {
                    printf("c_dir : %c ; c_LR : %c " , c_dir , c_LR);
                    for ( int i=0; i<length_P; i++ ) { printf( "%f " , Map_projection.at(key_pair).at(i) ); }
                    printf( "; " );
                    for ( int i=0; i<length_P; i++ ) { printf( "%f " , Map_A_inv_projection.at(key_pair).at(i) * dx ); }    
                    printf("\n");
                }
            }
            */

        } // set_forward_operators()


        //----------------------------------------------//
        //------------- Function prototype -------------//
        //----------------------------------------------//

        // #include "grid_drvt.tpp"

        #include "grid_update.tpp"

        // [2023/06/21]
        // NOTE: It seems that if we include the definition of the update () functions here
        //       in the header, the nvcc compiler may complain "calling __device__ function 
        //       from __host__ function is not allowed". 
        //           This is likely due to ns_type::host_precision, the precision type, being
        //       exposed to the nvcc compiler. However, if we template the update functions 
        //       with a generic type in its definition, the compilation seems to go through.
        //       The files are now organized in this way, with grid_update.tpp storing the 
        //       update function templates. 


        template<typename T>
        void fast2sum( T const a , T const b ,
                       T &     s , T &     t )
        {
              s = a + b;
            T z = s - a;
              t = b - z;
        }
        // [2023/06/21]
        // NOTE: This fast2sum () assumes b is less than or equal to a in magnitude, i.e., 
        //       it's the lower bits in b that gets lost in s and tracked in t. (No branch,
        //       i.e., if test, inside the function.)


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
        // [2023/06/21]
        // NOTE: slow2sum () is generally referred to as 2sum. A C++ identifier can't start 
        //       with a numeral. Using the name slow2sum () so that the function name would 
        //       have the same length as fast2sum (). 
        // [2023/09/18]
        // NOTE: Passing in a and b by value is important above. If s and t are references 
        //       to a and b, respectively (which is how we typically call this functions),
        //       then passing by reference would overwrite `a` erroneously.


        // [2023/06/16]
        // NOTE: In the following function template, cpst_N is the number of floating point 
        //       operations used to calculate the compensation/"lost bits"; cpst_N == 3 for 
        //       fast2sum; cpst_N == 6 for 2sum; cpst_S indicates where to store the "lost 
        //       bits"/compensation; 'C' incidates using a stand-alone memory variable; 'R' 
        //       indicates that the right hand side does double duty.
        //       
        //       /* In the fast2sum verison, we don't do if test, i.e., we assume that the 
        //          new item being added is smaller than the accumulating sum in magnitude. */
        //       
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

        // void adjust_parameters_energy ( ns_type::host_precision dx );
        // NOTE: The function adjust_parameters_energy () is removed so that we can focus on DO ONE THING WELL.

        void adjust_parameters_energy_periodic ();
        
        double energy_calculation ();

        template<typename T>
        void interpolate_forward_parameter (  
                                             run_time_vector< T > & forward_P );

        void retrieve_forward_parameter ( Class_Inverse_Specs &Inv_Specs );


};

#endif
