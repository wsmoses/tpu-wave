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
{}

__global__ void cuda_apply_source ( ns_type::cuda_precision * S , int ind , double increment )
{}

__global__ void cuda_record_soln ( ns_type::cuda_precision * R , ns_type::cuda_precision * S , int I_RCV )
{}

template<typename GridType>
__global__ void weighted_square_NORMAL_grid ( ns_type::cuda_precision * Sxx , ns_type::cuda_precision * Syy , 
                                              double * P1  , double * P2  , 
                                              double * T )
{}

template<typename GridType>
__global__ void weighted_square_SINGLE_grid ( ns_type::cuda_precision * S , 
                                              double * P , 
                                              double * T )
{}

template<int length>
__global__ void single_thread_reduce ( double * T , double * result )
{}

template<char C_type_x, char C_type_y, int C_size_x, int C_size_y, int C_chunk_size>
template<int N_block, int N_thread>
double cuda_Class_Grid<C_type_x, C_type_y, C_size_x, C_size_y, C_chunk_size>::energy_calculation ()
{
    return 0;
}


 

#endif