#ifndef GRID_CUH
#define GRID_CUH

#include <iostream>
#include <math.h>

#include "grid.hpp"

#include "namespace_type.cuh"
#include "cuda_container_run_time.cuh"

#include "namespace_device_variable.cuh"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>


// NOTE: cuda_Struct_Grid is intended to be passed as arguments for __global__ functions, which 
//       has a limit of 4KB and can only be passed by value, thus this struct needs to be small.
//       No member function should be allowed for the same token. 


// NOTE: On the other hand, cuda_Class_Grid is intended to be used on the host side to prepare
//       and organize data for kernel launch, we can be "liberal" with it.


struct cuda_Struct_Grid_Base
{
    // size of this grid (in terms of number of grid points)
    int G_size_x = -(1<<30); // 30 should be fine
    int G_size_y = -(1<<30); // 30 should be fine

    int chunk_size = 32;  // see cuda_grid_allocate_memory (); to round up the problem size when allocating memory                         
    
    int Lx_pad = -(1<<30); // 30 should be fine
    int Ly_pad = -(1<<30); // 30 should be fine

    int length_memory = -(1<<30);

    // stride: change in the 2D vector index caused by change in grid index in x or y direction by unit 1
    int stride_x;
    int stride_y;
};

template<char G_type_x = 'N', char G_type_y = 'N'>
struct cuda_Struct_Grid : public cuda_Struct_Grid_Base
{
    static constexpr char type_x = G_type_x;
    static constexpr char type_y = G_type_y;
};
// if "cuda_Struct_Grid" is intended to pass between host and device, we can remove the "cuda_" prefix from naming


class cuda_Class_Grid 
{
    public:

        Class_Grid * ptr_class_grid; 
        
        cuda_Struct_Grid_Base struct_grid;      

        // ------------------------------------------ //
        // ---- references to the struct members ---- //
        // ------------------------------------------ //

        char G_type_x;
        char G_type_y;

        int  & G_size_x      = struct_grid.G_size_x;
        int  & G_size_y      = struct_grid.G_size_y;

        int  & chunk_size    = struct_grid.chunk_size;  // used for calculating Lx_pad and Ly_pad

        int  & Lx_pad        = struct_grid.Lx_pad;
        int  & Ly_pad        = struct_grid.Ly_pad;

        int  & length_memory = struct_grid.length_memory;

        int  & stride_x      = struct_grid.stride_x;
        int  & stride_y      = struct_grid.stride_y;

        // ------------------------------------------ //
        // ---- references to the struct members ---- //
        // ------------------------------------------ //

        // [2023/06/21]
        // NOTE: Removed Map_G_type, Map_G_size, and Map_stride from this class since they are not used.
        //       If want to use aggregate type, std::array may be a better choice (particularly now that 
        //       we start to use std-20, which can make std::array constexpr).


        int N_modulo_x = 1<<30;  // Used to map the indices for periodic BC; needed when using OR not using the % operator
        int N_modulo_y = 1<<30;  // Used to map the indices for periodic BC; needed when using OR not using the % operator


        // pointers to the three grids that that this grid interacts with
        cuda_Class_Grid * pntr_Grid_x;
        cuda_Class_Grid * pntr_Grid_y;
        std::map< char , cuda_Class_Grid * > Map_pntr_grid;   // char : c_dir

        std::map< char , cuda_run_time_vector<ns_type::cuda_precision> * > Map_pntr_soln;
        
        std::vector< cuda_run_time_vector<ns_type::cuda_precision> > Vec_drvt;
        std::vector< cuda_run_time_vector<ns_type::cuda_precision> > Vec_soln;
        std::vector< cuda_run_time_vector<ns_type::cuda_precision> > Vec_prmt;
        
        std::vector< cuda_run_time_vector< double > > Vec_prmt_enrg;


        std::vector< cuda_run_time_vector<ns_type::cuda_precision> > Vec_cpst;
        std::vector< cuda_run_time_vector<ns_type::cuda_precision> > Vec_rths;

        
        cuda_run_time_vector<ns_type::cuda_precision> src_cpst;


        cuda_run_time_vector<double> thrust_memory;


        // ---- the number of derivative fields associated with this grid (always ns_forward::N_dir for the elastic case)
        int N_drvt = ns_forward::N_dir;
        // ---- the number of (wave)fields associated with this grid
        int N_soln = -1;         // 2 for NN ; 0 for MM ; 1 for the others
        // ---- the number of parameters associated with this grid
        int N_prmt = -1;         // 2 for NN ; 0 for MM ; 1 for the others
        // ---- the number of parameters associated with this grid
        int N_enrg = -1;         // 2 for NN ; 0 for MM ; 1 for the others

        
        std::string grid_name; 
        bool free_surface_update = false;    // set to true for Vx, Vy, Vz ; false for all stresses


        // ------ the stencil from ns_forward (with unit length) divided by dx used in this simulation
        cuda_run_time_vector<ns_type::cuda_precision> stencil_dt_dx;
        std::map< char , cuda_run_time_vector<int> > Map_stencil_shift;    // index difference when applying stencil to update the solution        


        // boundary operators from ns_forward (with unit length) divided by dx used in this simulation
        std::map< std::pair<char,char> , cuda_run_time_matrix<ns_type::cuda_precision> > Map_D_bdry;

        // 1D operators related to projection and its inverse operation (weighted by the diagonal components of the norm matrices) 
        // to weakly enforce the boundary conditions
        std::map< std::pair<char,char> , cuda_run_time_vector<ns_type::cuda_precision> > Map_projection;          // key : { c_dir , c_LR }
        std::map< std::pair<char,char> , cuda_run_time_vector<ns_type::cuda_precision> > Map_A_inv_projection;    // key : { c_dir , c_LR }
        std::map< std::pair<char,char> , cuda_run_time_vector<ns_type::cuda_precision> > Map_A_bdry_diag;         // key : { c_dir , c_LR }
        // Map_A_bdry_diag will be used in energy calculation (only)
    
        std::map< char , int > Map_interior_BGN;
        std::map< char , int > Map_interior_END;    // END stores past 1


        int SL_external;
        int SL_internal;


        std::map< std::pair<char,char> , int > Map_projection_BGN;
        std::map< std::pair<char,char> , int > Map_A_inv_projection_BGN;

        std::map< std::pair<char,char> , cuda_run_time_vector<ns_type::cuda_precision> > Map_buffer;


// the initialization of the following two fields should be included in this file

        
        cuda_run_time_vector_shallow_copy<ns_type::cuda_precision> stencil_dt_dx_shallow_copy;
        std::map< char , cuda_run_time_vector_shallow_copy<int> > Map_stencil_shift_shallow_copy;

        std::map< std::pair<char,char> , cuda_run_time_matrix_shallow_copy<ns_type::cuda_precision> > Map_D_bdry_shallow_copy;

        std::map< std::pair<char,char> , cuda_run_time_vector_shallow_copy<ns_type::cuda_precision> > Map_projection_shallow_copy;
        std::map< std::pair<char,char> , cuda_run_time_vector_shallow_copy<ns_type::cuda_precision> > Map_A_inv_projection_shallow_copy;
        std::map< std::pair<char,char> , cuda_run_time_vector_shallow_copy<ns_type::cuda_precision> > Map_A_bdry_diag_shallow_copy;


        std::vector< cuda_run_time_matrix<ns_type::cuda_precision> > Vec_RESULT_rcv;     // we probably want to change _matrix to _vector


        bool bool_stream = true;


        // struct Struct_Stream
        // {
        //     cudaStream_t soln;
        //     cudaStream_t drvt;
        // } struct_stream;  // this syntax works


        // std::map < std::string , cudaStream_t > Map_stream;
        // std::map < std::string , cudaStream_t > Map_event ;
        // if we use a map, mapping "dx_I" to stream, we can use for loop to create and destroy
        // we can also wrap both of them into a struct and define a constructor and destructor


        cudaStream_t stream_dx_I;
        cudaStream_t stream_dx_L;
        cudaStream_t stream_dx_R;

        cudaStream_t stream_bx_L;
        cudaStream_t stream_bx_R;

        cudaStream_t stream_dy;     // we may later separate this stream into 3 parts as for the x direction

        cudaStream_t stream_dy_I;     // we may later separate this stream into 3 parts as for the x direction
        cudaStream_t stream_dy_L;     // we may later separate this stream into 3 parts as for the x direction
        cudaStream_t stream_dy_R;     // we may later separate this stream into 3 parts as for the x direction

        cudaStream_t stream_soln;      // update, src, rcv can be put in this stream
        cudaStream_t stream_drvt;      // this one is for reset


        // constructor
        cuda_Class_Grid ( ) { }

        
        //-----------------------------------------------//
        //------------- Function defintiion -------------//
        //-----------------------------------------------//
        void cuda_Class_Grid_initialize ( Class_Grid * class_grid ) 
        {
            this->ptr_class_grid = class_grid;

            // NOTE: This function achieves set_grid_parameters () and set_forward_operators () 
            //       for the equivalent cpu class; We mostly copy from the equivalent cpu class, 
            //       except for containers that need to hold gpu memory, and _pad and stride_.

            this->G_type_x = class_grid->G_type_x;
            this->G_type_y = class_grid->G_type_y;


            this->G_size_x = class_grid->G_size_x;
            this->G_size_y = class_grid->G_size_y;


            // chunk_size set to 1 would make Lx_pad == G_size_x and Ly_pad == G_size_y
            this->Lx_pad = ( ( this->G_size_x + this->chunk_size - 1 ) / this->chunk_size ) * this->chunk_size;
            this->Ly_pad = ( ( this->G_size_y + this->chunk_size - 1 ) / this->chunk_size ) * this->chunk_size;

            this->length_memory = this->Lx_pad * this->Ly_pad;
            // [2023/06/21] NOTE: Can we use a better variable name than length_memory?

            this->stride_x = this->Ly_pad;
            this->stride_y =            1;


            this->N_modulo_x = class_grid->N_modulo_x;
            this->N_modulo_y = class_grid->N_modulo_y;


            this->SL_external = class_grid->SL_external;
            this->SL_internal = class_grid->SL_internal;


            this->grid_name           = class_grid->grid_name;
            this->free_surface_update = class_grid->free_surface_update;


            this->N_drvt = ns_forward::N_dir;
            this->N_soln = class_grid->N_soln;
            this->N_prmt = class_grid->N_prmt;
            this->N_enrg = class_grid->N_enrg;


            // allocate space for derivative, solution, and parameter fields
            if ( this->N_drvt > 0 ) { this->Vec_drvt     .reserve (this->N_drvt); }
            if ( this->N_soln > 0 ) { this->Vec_soln     .reserve (this->N_soln); }
            if ( this->N_prmt > 0 ) { this->Vec_prmt     .reserve (this->N_prmt); }

            if ( this->N_enrg > 0 ) { this->Vec_prmt_enrg.reserve (this->N_enrg); }

            if ( this->N_soln > 0 ) { this->Vec_cpst     .reserve (this->N_soln); }
            if ( this->N_soln > 0 ) { this->Vec_rths     .reserve (this->N_soln); }

            for ( int i = 0; i < this->N_drvt; i++ ) { this->Vec_drvt     .push_back ( cuda_run_time_vector<ns_type::cuda_precision> { this->length_memory } ); }
            for ( int i = 0; i < this->N_soln; i++ ) { this->Vec_soln     .push_back ( cuda_run_time_vector<ns_type::cuda_precision> { this->length_memory } ); }
            for ( int i = 0; i < this->N_prmt; i++ ) { this->Vec_prmt     .push_back ( cuda_run_time_vector<ns_type::cuda_precision> { this->length_memory } ); }
            
            for ( int i = 0; i < this->N_enrg; i++ ) { this->Vec_prmt_enrg.push_back ( cuda_run_time_vector< double > { this->length_memory } ); }

            for ( int i = 0; i < this->N_soln; i++ ) { this->Vec_cpst     .push_back ( cuda_run_time_vector<ns_type::cuda_precision> { this->length_memory } ); }
            for ( int i = 0; i < this->N_soln; i++ ) { this->Vec_rths     .push_back ( cuda_run_time_vector<ns_type::cuda_precision> { this->length_memory } ); }


            if ( this->N_enrg > 0 ) { this->thrust_memory.allocate_memory( this->length_memory ); }


            this->src_cpst.allocate_memory ( class_grid->N_soln );  // assume memory is set to zero            

           
			this->stencil_dt_dx.allocate_memory ( class_grid->stencil_dt_dx.length );
            this->stencil_dt_dx.copy_from_host <ns_type::host_precision> ( class_grid->stencil_dt_dx );

            // [2023/06/22]
            // NOTE: We should change copy_from_host to mem_copy_from_host and add a new function
            //       val_copy_from_host. The mem_copy_from_host version calls "cudaMemcpy", which
            //       requires the same data representation on host and device for correctness. The 
            //       val_copy_from_host version should copy in the unit of data item, and make the 
            //       appropriate conversion when necessary, which can be used in situations where 
            //       the host and device use different data formats, e.g., with double on host and 
            //       single (or half) on device, or with _Float16 on host and __half on device.


            for ( const char& c_dir : {'x','y'} )
            {
                this->Map_stencil_shift   [c_dir].allocate_memory ( class_grid->Map_stencil_shift.at(c_dir).length );
                this->Map_stencil_shift.at(c_dir).copy_from_host <int> ( class_grid->Map_stencil_shift.at(c_dir) );
            }
            // NOTE: Pay special attention to if the stencil_shift on cpu and gpu have the same 
            //       definition, i.e., are they multiplied by the strides already or not.

            // NOTE: We assume that x boundaries are weak; y is strong.
            for ( const char& c_dir : {'x'} )
            {
                for ( const char& c_LR : {'L','R'} )
                {
                    auto & host_D_bdry = class_grid->Map_D_bdry.at( { c_dir , c_LR } );

                    this->Map_D_bdry   [ { c_dir , c_LR } ].allocate_memory ( host_D_bdry.rows , host_D_bdry.cols );
                    this->Map_D_bdry.at( { c_dir , c_LR } ).copy_from_host <ns_type::host_precision> ( host_D_bdry );


                    auto & host_projection = class_grid->Map_projection.at( { c_dir , c_LR } );

                    this->Map_projection   [ { c_dir , c_LR } ].allocate_memory ( host_projection.length );
                    this->Map_projection.at( { c_dir , c_LR } ).copy_from_host <ns_type::host_precision> ( host_projection );


                    auto & host_A_inv_projection = class_grid->Map_A_inv_projection.at( { c_dir , c_LR } );

                    this->Map_A_inv_projection   [ { c_dir , c_LR } ].allocate_memory ( host_A_inv_projection.length );
                    this->Map_A_inv_projection.at( { c_dir , c_LR } ).copy_from_host <ns_type::host_precision> ( host_A_inv_projection );


                    auto & host_A_bdry_diag = class_grid->Map_A_bdry_diag.at( { c_dir , c_LR } );

                    this->Map_A_bdry_diag   [ { c_dir , c_LR } ].allocate_memory ( host_A_bdry_diag.length );
                    this->Map_A_bdry_diag.at( { c_dir , c_LR } ).copy_from_host <ns_type::host_precision> ( host_A_bdry_diag );
                }
            }


            // NOTE: We assume that x boundaries are weak; y is strong.
            for ( const char & c_dir : {'x'} )
            {
                for ( const char & c_LR : {'L','R'} )
                {
                    int buffer_length = -(1<<30);
                    if ( c_dir == 'x' ) { buffer_length = this->Ly_pad; }
                    if ( c_dir == 'y' ) { buffer_length = this->Lx_pad; }                    

                    this->Map_buffer   [ { c_dir , c_LR } ].allocate_memory ( buffer_length );
                    this->Map_buffer.at( { c_dir , c_LR } ).memset_zero ();
                }
            }


            this->Map_interior_BGN = class_grid->Map_interior_BGN;
            this->Map_interior_END = class_grid->Map_interior_END;


            this->stencil_dt_dx.make_shallow_copy ( stencil_dt_dx_shallow_copy );

            for ( const auto & iter_map : Map_stencil_shift ) 
                { iter_map.second.make_shallow_copy ( Map_stencil_shift_shallow_copy [iter_map.first] ); }

            for ( const auto & iter_map : Map_D_bdry ) 
                { iter_map.second.make_shallow_copy ( Map_D_bdry_shallow_copy [iter_map.first] ); }

            for ( const auto & iter_map : Map_projection ) 
                { iter_map.second.make_shallow_copy ( Map_projection_shallow_copy [iter_map.first] ); }

            for ( const auto & iter_map : Map_A_inv_projection ) 
                { iter_map.second.make_shallow_copy ( Map_A_inv_projection_shallow_copy [iter_map.first] ); }
            
            for ( const auto & iter_map : Map_A_bdry_diag ) 
                { iter_map.second.make_shallow_copy ( Map_A_bdry_diag_shallow_copy [iter_map.first] ); }

        } // cuda_Class_Grid_initialize()


        //-----------------------------------------------//
        //------------- Function defintiion -------------//
        //-----------------------------------------------//
        void set_grid_pointers ( std::map< std::array<char, ns_forward::N_dir> , cuda_Class_Grid * > & Map_cuda_grid_pointers )
        {
            char P_type_x = ( G_type_x == 'N' ? 'M' : 'N' );
            char P_type_y = ( G_type_y == 'N' ? 'M' : 'N' );

            this->pntr_Grid_x = Map_cuda_grid_pointers.at( { P_type_x , G_type_y } ) ; 
            this->pntr_Grid_y = Map_cuda_grid_pointers.at( { G_type_x , P_type_y } ) ; 

            Map_pntr_grid['x'] = pntr_Grid_x;
            Map_pntr_grid['y'] = pntr_Grid_y;


            // check the sizes
            if ( G_type_x == 'N' ) assert ( G_size_x - pntr_Grid_x->G_size_x ==  1 );
            if ( G_type_x == 'M' ) assert ( G_size_x - pntr_Grid_x->G_size_x == -1 );

            if ( G_type_y == 'N' ) assert ( G_size_y - pntr_Grid_y->G_size_y ==  1 );
            if ( G_type_y == 'M' ) assert ( G_size_y - pntr_Grid_y->G_size_y == -1 );


            // assign to pointers the solution variables on the interacting grids
            if ( pntr_Grid_x->N_soln == 2 ) { Map_pntr_soln['x'] = & pntr_Grid_x->Vec_soln.at(0); }
            if ( pntr_Grid_y->N_soln == 2 ) { Map_pntr_soln['y'] = & pntr_Grid_y->Vec_soln.at(1); }

            if ( pntr_Grid_x->N_soln == 1 ) { Map_pntr_soln['x'] = & pntr_Grid_x->Vec_soln.at(0); }
            if ( pntr_Grid_y->N_soln == 1 ) { Map_pntr_soln['y'] = & pntr_Grid_y->Vec_soln.at(0); }

        } // set_grid_pointers()


        //-----------------------------------------------//
        //------------- Function defintiion -------------//
        //-----------------------------------------------//
        void copy_field_pitched ( std::string field_name , int ind_field , std::string copy_direction )
        {
            ns_type::cuda_precision * dev_ptr = nullptr;

                 if ( strcmp( field_name.c_str(), "drvt" ) == 0 ) { dev_ptr = this->Vec_drvt     .at(ind_field).ptr; }
            else if ( strcmp( field_name.c_str(), "soln" ) == 0 ) { dev_ptr = this->Vec_soln     .at(ind_field).ptr; }
            else if ( strcmp( field_name.c_str(), "prmt" ) == 0 ) { dev_ptr = this->Vec_prmt     .at(ind_field).ptr; }
            else 
                { printf ("%s %d Unrecognized field name.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }


            std::vector< ns_type::host_precision * > ptr_hst_fields;

            if ( strcmp( field_name.c_str(), "drvt" ) == 0 ) { for ( auto & iter : ptr_class_grid->Map_v_D       ) { ptr_hst_fields.push_back ( iter.second.ptr ); } }
            if ( strcmp( field_name.c_str(), "soln" ) == 0 ) { for ( auto & iter : ptr_class_grid->Vec_soln      ) { ptr_hst_fields.push_back ( iter.ptr );        } }
            if ( strcmp( field_name.c_str(), "prmt" ) == 0 ) { for ( auto & iter : ptr_class_grid->Vec_prmt      ) { ptr_hst_fields.push_back ( iter.ptr );        } }

            if ( strcmp( field_name.c_str(), "drvt" ) == 0 ) { assert ( (int) ptr_hst_fields.size() == this->N_drvt ); } 
            if ( strcmp( field_name.c_str(), "soln" ) == 0 ) { assert ( (int) ptr_hst_fields.size() == this->N_soln ); } 
            if ( strcmp( field_name.c_str(), "prmt" ) == 0 ) { assert ( (int) ptr_hst_fields.size() == this->N_prmt ); } 
            
            ns_type::host_precision * hst_ptr = ptr_hst_fields.at(ind_field);


            // assert ( sizeof(ns_type::cuda_precision) == sizeof(ns_type::host_precision) );
            // [2023/05/13]
            // NOTE: If setting both "host_precision" and "cuda_precision" to "long double", the
            //       nvcc compiler warns that "long double" is treated as "double" in device code.
            //       But maybe on host, it is still the 80 bit version and "compatible" with type 
            //       host_precision? And that's why the above assert passes? 
            // [2023/05/14]
            // NOTE: Actually, both prints out 16 bytes.
            // [2023/06/25]
            // NOTE: It makes sense that the 80 bit version is stored as 16 bytes (for alignment 
            //       purpose). 
            // printf( "\n\n%s %d cuda_precision: %d, host_precision: %d\n\n", __FILE__, __LINE__, 
            //          sizeof(ns_type::cuda_precision), sizeof(ns_type::host_precision) );
            // [2023/06/25]
            // NOTE: "sizeof ()" is not a good test for if the two types are the same. "is_same_v" 
            //       is better.
            static_assert ( std::is_same_v< ns_type::cuda_precision , ns_type::host_precision > , "Not the same type." );

            if ( strcmp( copy_direction.c_str(), "hst_to_dev" ) == 0 )
            {
                cudaMemcpy2D( dev_ptr, Ly_pad   * sizeof(ns_type::cuda_precision), 
                              hst_ptr, G_size_y * sizeof(ns_type::host_precision),
                              G_size_y * sizeof(ns_type::host_precision) , 
                              G_size_x , cudaMemcpyHostToDevice );
            }
            else if ( strcmp( copy_direction.c_str(), "dev_to_hst" ) == 0 )
            {
                cudaMemcpy2D( hst_ptr, G_size_y * sizeof(ns_type::host_precision),
                              dev_ptr, Ly_pad   * sizeof(ns_type::cuda_precision), 
                              G_size_y * sizeof(ns_type::host_precision) , 
                              G_size_x , cudaMemcpyDeviceToHost );
            }
            else 
                { printf ("%s %d Unrecognized direction.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
        }


        //-----------------------------------------------//
        //------------- Function defintiion -------------//
        //-----------------------------------------------//
        void copy_field_pitched_enrg ( std::string field_name , int ind_field , std::string copy_direction )
        {
            double * dev_ptr = nullptr;

                 if ( strcmp( field_name.c_str(), "enrg" ) == 0 ) { dev_ptr = this->Vec_prmt_enrg.at(ind_field).ptr; } 
            else 
                { printf ("%s %d Unrecognized field name.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }


            std::vector< double * > ptr_hst_fields;

            if ( strcmp( field_name.c_str(), "enrg" ) == 0 ) { for ( auto & iter : ptr_class_grid->Vec_prmt_enrg ) { ptr_hst_fields.push_back ( iter.ptr );        } }
            
            if ( strcmp( field_name.c_str(), "enrg" ) == 0 ) { assert ( (int) ptr_hst_fields.size() == this->N_enrg ); } 
            
            double * hst_ptr = ptr_hst_fields.at(ind_field);


            if ( strcmp( copy_direction.c_str(), "hst_to_dev" ) == 0 )
            {
                cudaMemcpy2D( dev_ptr, Ly_pad   * sizeof(double), 
                              hst_ptr, G_size_y * sizeof(double),
                              G_size_y * sizeof(double) , 
                              G_size_x , cudaMemcpyHostToDevice );
            }
            else if ( strcmp( copy_direction.c_str(), "dev_to_hst" ) == 0 )
            {
                cudaMemcpy2D( hst_ptr, G_size_y * sizeof(double),
                              dev_ptr, Ly_pad   * sizeof(double), 
                              G_size_y * sizeof(double) , 
                              G_size_x , cudaMemcpyDeviceToHost );
            }
            else 
                { printf ("%s %d Unrecognized direction.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
        }


        //-----------------------------------------------//
        //------------- Function defintiion -------------//
        //-----------------------------------------------//
        void reset_field ( std::string field_name , int ind_field )
        {
            ns_type::cuda_precision * dev_ptr = nullptr;

                 if ( strcmp( field_name.c_str(), "drvt" ) == 0 ) { dev_ptr = this->Vec_drvt.at(ind_field).ptr; }
            else if ( strcmp( field_name.c_str(), "soln" ) == 0 ) { dev_ptr = this->Vec_soln.at(ind_field).ptr; }
            else if ( strcmp( field_name.c_str(), "prmt" ) == 0 ) { dev_ptr = this->Vec_prmt.at(ind_field).ptr; }
            else 
                { printf ("%s %d Unrecognized field name.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            auto & stream = this->stream_drvt;

            if ( bool_stream == false )
                { cudaMemset      ( dev_ptr, 0, this->length_memory * sizeof(ns_type::cuda_precision) ); }
            else
                { cudaMemsetAsync ( dev_ptr, 0, this->length_memory * sizeof(ns_type::cuda_precision), stream ); }
        }


        // NOTE: If the data structure are consistent between host and device fields, things would be so much simpler.
        // [2023/07/13] 
        // NOTE: By "consistent" above, I think we mean they have the same padding (0 is a possibility).
        //       
        // NOTE: The following member function has not been tested. 
        // [2023/07/13]
        // NOTE: The actual copying (mem_ or val_) is wrapped inside the copy_from_host and copy_to_host member functions 
        //       of cuda_run_time_vector. 
        //-----------------------------------------------//
        //------------- Function defintiion -------------//
        //-----------------------------------------------//
        void copy_field_contiguous ( std::string field_name , int ind_field , std::string copy_direction )
        {
            cuda_run_time_vector<ns_type::cuda_precision> * dev_vec = nullptr;

                 if ( strcmp( field_name.c_str(), "drvt" ) == 0 ) { dev_vec = & this->Vec_drvt.at(ind_field); }
            else if ( strcmp( field_name.c_str(), "soln" ) == 0 ) { dev_vec = & this->Vec_soln.at(ind_field); }
            else if ( strcmp( field_name.c_str(), "prmt" ) == 0 ) { dev_vec = & this->Vec_prmt.at(ind_field); }
            else 
                { printf ("%s %d Unrecognized field name.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            std::vector< run_time_vector<ns_type::host_precision> * > vec_hst_fields;

            if ( strcmp( field_name.c_str(), "drvt" ) == 0 ) { for ( auto & iter : ptr_class_grid->Map_v_D  ) { vec_hst_fields.push_back ( & iter.second ); } }
            if ( strcmp( field_name.c_str(), "soln" ) == 0 ) { for ( auto & iter : ptr_class_grid->Vec_soln ) { vec_hst_fields.push_back ( & iter );        } } 
            if ( strcmp( field_name.c_str(), "prmt" ) == 0 ) { for ( auto & iter : ptr_class_grid->Vec_prmt ) { vec_hst_fields.push_back ( & iter );        } }

            if ( strcmp( field_name.c_str(), "drvt" ) == 0 ) { assert ( (int) vec_hst_fields.size() == this->N_drvt ); } 
            if ( strcmp( field_name.c_str(), "soln" ) == 0 ) { assert ( (int) vec_hst_fields.size() == this->N_soln ); } 
            if ( strcmp( field_name.c_str(), "prmt" ) == 0 ) { assert ( (int) vec_hst_fields.size() == this->N_prmt ); } 
            
            run_time_vector<ns_type::host_precision> * hst_vec = vec_hst_fields.at(ind_field);

                 if ( strcmp( copy_direction.c_str(), "hst_to_dev" ) == 0 ) { dev_vec->copy_from_host <ns_type::host_precision> ( * hst_vec ); }
            else if ( strcmp( copy_direction.c_str(), "dev_to_hst" ) == 0 ) { dev_vec->copy_to_host   <ns_type::host_precision> ( * hst_vec ); }
            else 
                { printf ("%s %d Unrecognized direction.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
        }


        // [2022/07/18]
        // NOTE: In the following function names, _x and _y refer to the "data" direction, 
        //       not the "physical" direction.
        template<bool bool_extra=false, int N_block=600, int N_thread=32>
        void kernel_launch_cuda_interior_x ();
        
        template<bool bool_extra=false, int N_block=600, int N_thread=32>
        void kernel_launch_cuda_boundary_x ( char c_LR );

        template<bool bool_extra=false, int N_block=1, int N_thread=32>
        void kernel_launch_cuda_secure_bdry_x ( char c_LR );

        template<bool bool_extra=false, char soln_type_x='N', char soln_type_y='N', int N_block=600, int N_thread=32>
        void kernel_launch_cuda_periodic_y_modulo ();

        template<int cpst_N = 0 , char cpst_S = '0', int N_block=600, int N_thread=32>
        void kernel_launch_cuda_update ();

        template<int N_block=600, int N_thread=32>
        double energy_calculation ();
};


template<bool bool_extra=false>
__device__ void drvt_to_rths ( ns_type::cuda_precision v_d , int i_d , 
                               ns_type::cuda_precision * P ,
                               ns_type::cuda_precision * P_extra ,
                               ns_type::cuda_precision * R ,
                               ns_type::cuda_precision * R_extra )
{
    // add derivative to rhs
    R[i_d] += v_d * P[i_d];

    if constexpr ( bool_extra )
    { 
        R_extra[i_d] += v_d * P      [i_d];
        R_extra[i_d] += v_d * P_extra[i_d];
    }
    // [2023/07/15]
    // NOTE: The meaning of R and R_extra may be slightly different from the cpu code.
    //       We probably should change the cpu code so that they are consistent.
}
// [2023/07/27]
// NOTE: When using C++17, we need to place "drvt_to_rths ()" before the following 
//       "#include"s because it is called therein. Alternatively, we could use a 
//       forward declaration for "drvt_to_rths ()".


#include "grid_kernel_x.cut"

#include "grid_kernel_y.cut"

#include "grid_update.cut"


// [2023/07/17]
// NOTE: We really should use either a template parameter or static constepxr boolean to 
//       indicate if a grid is NORMAL or SINGLE - it will simplify the implementation a 
//       lot; static constexpr may involve less code change. (Need to check if a static
//       constexpr variable of a regular class can be accessed at compile time.)


__global__ void cuda_apply_source ( ns_type::cuda_precision * S , int ind , 
                                    ns_type::cuda_precision * R , int i_field ,
                                    double increment );

__global__ void cuda_apply_source ( ns_type::cuda_precision * S , int ind , double increment );

__global__ void cuda_record_soln ( ns_type::cuda_precision * R , ns_type::cuda_precision * S , int I_RCV );

__global__ void cuda_print_soln ( ns_type::cuda_precision * S , int ind , int it );


__global__ void weighted_square_NORMAL_grid ( cuda_Struct_Grid struct_grid , 
                                              ns_type::cuda_precision * Sxx , ns_type::cuda_precision * Syy  , 
                                              double * P1  , double * P2   , 
                                              double * T );  // T stores the intermediate output 

__global__ void weighted_square_SINGLE_grid ( cuda_Struct_Grid struct_grid , 
                                              ns_type::cuda_precision * S , double * P , double * T );  // T stores the intermediate output 

#endif