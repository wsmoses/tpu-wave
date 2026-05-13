#ifndef GRID_CUH
#define GRID_CUH

#include <iostream>
#include <math.h>

#include "grid.hpp"

#include "namespace_type.cuh"
#include "cuda_container_run_time.cuh"

#include "namespace_device_variable.cuh"




// NOTE: cuda_Struct_Grid is intended to be passed as arguments for __global__ functions, which 
//       has a limit of 4KB and can only be passed by value, thus this struct needs to be small.
//       No member function should be allowed for the same token. 


// NOTE: On the other hand, cuda_Class_Grid is intended to be used on the host side to prepare
//       and organize data for kernel launch, we can be "liberal" with it.


template<char G_type_x, char G_type_y, int G_size_x, int G_size_y, int chunk_size = 32>
struct cuda_Struct_Grid
{
    static constexpr char type_x = G_type_x;
    static constexpr char type_y = G_type_y;
    static constexpr int size_x = G_size_x;
    static constexpr int size_y = G_size_y;
    static constexpr int chunk = chunk_size;

    static constexpr int Lx_pad = ((G_size_x + chunk_size - 1) / chunk_size) * chunk_size;
    static constexpr int Ly_pad = ((G_size_y + chunk_size - 1) / chunk_size) * chunk_size;

    static constexpr int length_memory = Lx_pad * Ly_pad;

    static constexpr int stride_x = Ly_pad;
    static constexpr int stride_y = 1;
};
// if "cuda_Struct_Grid" is intended to pass between host and device, we can remove the "cuda_" prefix from naming


class cuda_Class_Grid_Base 
{
    public:
        Class_Grid * ptr_class_grid; 
        
        char G_type_x;
        char G_type_y;

        int G_size_x;
        int G_size_y;
        int chunk_size;
        int Lx_pad;
        int Ly_pad;
        int length_memory;
        int stride_x;
        int stride_y;

        cuda_Class_Grid_Base(char tx, char ty, int sx, int sy, int chunk, int lx, int ly, int len, int strx, int stry) 
            : G_type_x(tx), G_type_y(ty), G_size_x(sx), G_size_y(sy), chunk_size(chunk), Lx_pad(lx), Ly_pad(ly), length_memory(len), stride_x(strx), stride_y(stry) {}


        int N_modulo_x = 1<<30;  // Used to map the indices for periodic BC; needed when using OR not using the % operator
        int N_modulo_y = 1<<30;  // Used to map the indices for periodic BC; needed when using OR not using the % operator


        // pointers to the three grids that that this grid interacts with
        cuda_Class_Grid_Base * pntr_Grid_x;
        cuda_Class_Grid_Base * pntr_Grid_y;
        std::map< char , cuda_Class_Grid_Base * > Map_pntr_grid;   // char : c_dir

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






        virtual ~cuda_Class_Grid_Base() {}

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
                if ( Ly_pad == G_size_y ) {
                    cudaMemcpy( dev_ptr, hst_ptr, G_size_x * G_size_y * sizeof(ns_type::cuda_precision), cudaMemcpyHostToDevice );
                } else {
                    for ( int i = 0; i < G_size_x; i++ ) {
                        cudaMemcpy( dev_ptr + i * Ly_pad, 
                                    hst_ptr + i * G_size_y, 
                                    G_size_y * sizeof(ns_type::cuda_precision), 
                                    cudaMemcpyHostToDevice );
                    }
                }
            }
            else if ( strcmp( copy_direction.c_str(), "dev_to_hst" ) == 0 )
            {
                if ( Ly_pad == G_size_y ) {
                    cudaMemcpy( hst_ptr, dev_ptr, G_size_x * G_size_y * sizeof(ns_type::host_precision), cudaMemcpyDeviceToHost );
                } else {
                    for ( int i = 0; i < G_size_x; i++ ) {
                        cudaMemcpy( hst_ptr + i * G_size_y, 
                                    dev_ptr + i * Ly_pad, 
                                    G_size_y * sizeof(ns_type::host_precision), 
                                    cudaMemcpyDeviceToHost );
                    }
                }
            }
            else 
                { printf ("%s %d Unrecognized direction.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
        }

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
                if ( Ly_pad == G_size_y ) {
                    cudaMemcpy( dev_ptr, hst_ptr, G_size_x * G_size_y * sizeof(double), cudaMemcpyHostToDevice );
                } else {
                    for ( int i = 0; i < G_size_x; i++ ) {
                        cudaMemcpy( dev_ptr + i * Ly_pad, 
                                    hst_ptr + i * G_size_y, 
                                    G_size_y * sizeof(double), 
                                    cudaMemcpyHostToDevice );
                    }
                }
            }
            else if ( strcmp( copy_direction.c_str(), "dev_to_hst" ) == 0 )
            {
                if ( Ly_pad == G_size_y ) {
                    cudaMemcpy( hst_ptr, dev_ptr, G_size_x * G_size_y * sizeof(double), cudaMemcpyDeviceToHost );
                } else {
                    for ( int i = 0; i < G_size_x; i++ ) {
                        cudaMemcpy( hst_ptr + i * G_size_y, 
                                    dev_ptr + i * Ly_pad, 
                                    G_size_y * sizeof(double), 
                                    cudaMemcpyDeviceToHost );
                    }
                }
            }
            else 
                { printf ("%s %d Unrecognized direction.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
        }

        virtual void cuda_Class_Grid_initialize ( Class_Grid * class_grid ) = 0;
        virtual void set_grid_pointers ( std::map< std::array<char, ns_forward::N_dir> , cuda_Class_Grid_Base * > & Map_cuda_grid_pointers ) = 0;
};


template<char C_type_x = 'N', char C_type_y = 'N', int C_size_x = 600, int C_size_y = 600, int C_chunk_size = 32>
class cuda_Class_Grid : public cuda_Class_Grid_Base
{
    public:
        using GridStruct = cuda_Struct_Grid<C_type_x, C_type_y, C_size_x, C_size_y, C_chunk_size>;

        GridStruct my_struct_grid;      

        static constexpr char type_x = C_type_x;
        static constexpr char type_y = C_type_y;

        cuda_Class_Grid() : cuda_Class_Grid_Base(C_type_x, C_type_y, C_size_x, C_size_y, C_chunk_size, GridStruct::Lx_pad, GridStruct::Ly_pad, GridStruct::length_memory, GridStruct::stride_x, GridStruct::stride_y) {}

        // constructor
        // cuda_Class_Grid ( ) { }
        // [2024/04/03] NOTE: commented out since we added a constructor above.
        void cuda_Class_Grid_initialize ( Class_Grid * class_grid ) override
        {
            this->ptr_class_grid = class_grid;

            // NOTE: This function achieves set_grid_parameters () and set_forward_operators () 
            //       for the equivalent cpu class; We mostly copy from the equivalent cpu class, 
            //       except for containers that need to hold gpu memory, and _pad and stride_.

            assert(type_x == class_grid->G_type_x);
            assert(type_y == class_grid->G_type_y);

            assert(GridStruct::size_x == class_grid->G_size_x);
            assert(GridStruct::size_y == class_grid->G_size_y);

            // Pads and strides are already initialized in struct_grid constructor.


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

            for ( int i = 0; i < this->N_soln; i++ ) { this->Vec_soln     .push_back ( cuda_run_time_vector<ns_type::cuda_precision> { this->length_memory } ); } 
            for ( int i = 0; i < this->N_enrg; i++ ) { this->Vec_prmt_enrg.push_back ( cuda_run_time_vector< double > { this->length_memory } ); }

	    this->thrust_memory.allocate_memory( this->length_memory );

        } // cuda_Class_Grid_initialize()


        //-----------------------------------------------//
        //------------- Function defintiion -------------//
        //-----------------------------------------------//
        void set_grid_pointers ( std::map< std::array<char, ns_forward::N_dir> , cuda_Class_Grid_Base * > & Map_cuda_grid_pointers ) override
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
        void reset_field ( std::string field_name , int ind_field )
        {
            ns_type::cuda_precision * dev_ptr = nullptr;

                 if ( strcmp( field_name.c_str(), "drvt" ) == 0 ) { dev_ptr = this->Vec_drvt.at(ind_field).ptr; }
            else if ( strcmp( field_name.c_str(), "soln" ) == 0 ) { dev_ptr = this->Vec_soln.at(ind_field).ptr; }
            else if ( strcmp( field_name.c_str(), "prmt" ) == 0 ) { dev_ptr = this->Vec_prmt.at(ind_field).ptr; }
            else 
                { printf ("%s %d Unrecognized field name.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            cudaMemsetAsync ( dev_ptr, 0, this->length_memory * sizeof(ns_type::cuda_precision), 0 );
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
                                    double increment )
{
    R [ i_field ] += increment;
    device_slow2sum <ns_type::cuda_precision> ( S[ind] , R[i_field] , S[ind] , R[i_field] );
}

__global__ void cuda_apply_source ( ns_type::cuda_precision * S , int ind , double increment )
{
    S [ ind ] += increment;
}

__global__ void cuda_record_soln ( ns_type::cuda_precision * R , ns_type::cuda_precision * S , int I_RCV )
{
    * R = S [ I_RCV ];
}

// __global__ void cuda_print_soln ( ns_type::cuda_precision * S , int ind , int it )
// {
//     printf( "Time step: %6d soln: % 12.11e \n", it, (double) S [ind] );
// }


// NOTE: Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-parameters
//
//     __global__ function parameters are PASSED TO THE DEVICE via constant memory and are LIMITED TO 4 KB.
//     __global__ functions cannot have a variable number of arguments.
//     __global__ function parameters CANNOT BE PASS-BY-REFERENCE.
//
// Since there is a limit of 4KB (i.e., 512 doubles), let's make a struct that contains only data members.


template<typename GridType>
__global__ void weighted_square_NORMAL_grid ( ns_type::cuda_precision * Sxx , ns_type::cuda_precision * Syy , 
                                              double * P1  , double * P2  , 
                                              double * T )  // T stores the intermediate output 
{    
    int ind_block  =  blockIdx.x;
    int ind_thread = threadIdx.x;

    int ix = ind_block;
    if ( ix >= GridType::size_x ) return;

    {
        for ( int iy = 0; iy < GridType::size_y; iy += blockDim.x )
        {            
            int actual_iy = iy + ind_thread;
            if (actual_iy < GridType::size_y) {
                int ind = ix * GridType::Ly_pad + actual_iy;

                T [ind]  = (double) Sxx [ind] * P1 [ind]
                         + (double) Sxx [ind] * P2 [ind] * (double) Syy [ind];
            }
        }
    }
}


template<typename GridType>
__global__ void weighted_square_SINGLE_grid ( ns_type::cuda_precision * S , 
                                              double * P , 
                                              double * T )  // T stores the intermediate output 
{    
    int ind_block  =  blockIdx.x;
    int ind_thread = threadIdx.x;

    int ix = ind_block;
    if ( ix >= GridType::size_x ) return;

    {
        for ( int iy = 0; iy < GridType::size_y; iy += blockDim.x )
        {            
            int actual_iy = iy + ind_thread;
            if (actual_iy < GridType::size_y) {
                int ind = ix * GridType::Ly_pad + actual_iy;
                T [ind] = S [ind] * P [ind];
            }
        }
    }
}


template<int length>
__global__ void single_thread_reduce ( double * T , double * result )
{
    double sum = 0;
    for ( int i = 0; i < length; i++ ) {
        sum += T[i];
    }
    *result = sum;
}

template<char C_type_x, char C_type_y, int C_size_x, int C_size_y, int C_chunk_size>
template<int N_block, int N_thread>
double cuda_Class_Grid<C_type_x, C_type_y, C_size_x, C_size_y, C_chunk_size>::energy_calculation ()
{
    auto & T = this->thrust_memory;

    if ( strcmp( grid_name.c_str(), "SMM" ) == 0 )
    {
        auto & Sxx = Vec_soln .at(0);
        auto & Syy = Vec_soln .at(1);

        auto & P1  = Vec_prmt_enrg.at(0);
        auto & P2  = Vec_prmt_enrg.at(1);

        weighted_square_NORMAL_grid <GridStruct> <<< N_block , N_thread >>> ( Sxx.ptr , Syy.ptr , 
                                                                 P1 .ptr , P2 .ptr , 
                                                                 T  .ptr );
    }
    else
    {
        auto & S = Vec_soln .at(0);
        auto & P = Vec_prmt_enrg.at(0);        
        auto & T = this->thrust_memory;

        weighted_square_SINGLE_grid <GridStruct> <<< N_block , N_thread >>> ( S.ptr , P.ptr , T.ptr );
    }

    double * d_result = nullptr;
    cudaMalloc( &d_result , sizeof(double) );
    single_thread_reduce <GridStruct::length_memory> <<< 1 , 1 >>> ( T.ptr , d_result );
    
    double E = 0;
    cudaMemcpy( &E , T.ptr , sizeof(double) , cudaMemcpyDeviceToHost );
    cudaFree( d_result );

    return E/2.;
}


 

#endif
