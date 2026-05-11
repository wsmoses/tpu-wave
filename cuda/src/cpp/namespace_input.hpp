#ifndef NAMESPACE_INPUT_H
#define NAMESPACE_INPUT_H

#include <array>
#include <math.h>
#include <assert.h>

#include "namespace_forward.hpp"
#include "container_run_time.hpp"

namespace namespace_input
{
    extern std::string medium_name;

    extern std::array<char,ns_forward::N_dir> bdry_type_L;
    extern std::array<char,ns_forward::N_dir> bdry_type_R;

    extern std::map< std::pair<char,char> , char > Map_bdry_type;

    extern std::array<int, ns_forward::N_dir> prmt_M_sizes;
    extern std::array<int, ns_forward::N_dir> soln_M_sizes;

    extern long Mx_soln, Nx_soln;
    extern long My_soln, Ny_soln;

    extern long Mx_prmt, Nx_prmt;
    extern long My_prmt, Ny_prmt;

    extern long inv_prmt_stride_x;
    extern long inv_prmt_stride_y;

    extern long        inv_prmt_size;
    extern long PADDED_inv_prmt_size;

    // [2023/01/11]
    // NOTE: Now that we have inlcuded Mx_soln, Nx_soln, My_soln, Ny_soln and 
    //       Mx_prmt, Nx_prmt, My_prmt, Ny_prmt as inputs here, do we still 
    //       need prmt_M_sizes and soln_M_sizes as inputs?

    extern int num_dx_prmt;
    extern int den_dx_prmt;
    
    extern int num_dx_soln;
    extern int den_dx_soln;

    extern int    Nt;
    extern double dt_max;
    extern double CFL_constant;

    extern double central_f;
    extern double time_delay;

    extern char c_energy;
    extern bool bool_energy;

    extern bool bool_visual;

    extern double dx;
    extern double dt;

    extern double source_scaling;

    extern int  device_number;

    extern run_time_vector<double> Record_E_k;
    extern run_time_vector<double> Record_E_p0;
    extern run_time_vector<double> Record_E_p1;
    // [2023/06/09]
    // NOTE: Only two of them are needed.
    // [2023/06/10]
    // NOTE: Do we want to keep them here or move them to Inv_Specs?
}
namespace ns_input = namespace_input;


inline void print_discretization_parameters ( double const min_velocity , double const max_velocity )
{
    using namespace ns_input;
    using ns_forward::N_dir;

    printf("\n");
    printf(" dx: %10.7f", dx);                            printf("%*c    ", 2, ' ');

    // printf(" total length (x): %10.7f", dx * Mx_soln);    printf("%*c    ", 2, ' ');
    // printf(" total length (y): %10.7f", dx * My_soln);    printf("%*c    ", 2, ' ');

    printf(" total length (x): %10.7f", dx * soln_M_sizes.at(0));    printf("%*c    ", 2, ' ');
    printf(" total length (y): %10.7f", dx * soln_M_sizes.at(1));    printf("%*c    ", 2, ' ');


    // [2023/01/15]
    // NOTE: We probably want to directly specify Mx_soln, My_soln, Mx_prmt, My_prmt
    //       and remove prmt_M_sizes and soln_M_sizes.

    
    printf(" dt: %10.7f", dt);                            printf("%*c    ", 2, ' ');
    printf(" total time: %10.7f", dt*Nt);                 printf("%*c  \n", 2, ' ');
    

    double min_lambda      = min_velocity / ( central_f * 2.5 );

    double effective_N_ppw = min_lambda / dx;
    printf("effective N_ppw:        %10.7f (min_velocity: %10.7f)\n", 
            effective_N_ppw,                min_velocity); 

    double effective_CFL_constant = ( dt / dx ) * ( sqrt( (double) N_dir ) * max_velocity );
    printf("effective CFL_constant: %10.7f (max_velocity: %10.7f)\n", 
            effective_CFL_constant,         max_velocity); 

    fflush(stdout);
}


inline void ns_input_derived_variables ()
{
    using namespace ns_input;

    // ---- inv

    Mx_prmt = prmt_M_sizes.at(0);  Nx_prmt = Mx_prmt + 1;
    My_prmt = prmt_M_sizes.at(1);  Ny_prmt = My_prmt + 1;

    inv_prmt_stride_x = Ny_prmt;
    inv_prmt_stride_y = 1;

           inv_prmt_size =   Nx_prmt *         Ny_prmt;
    PADDED_inv_prmt_size = ( Nx_prmt + 1 ) * ( Ny_prmt + 1 );

    if ( inv_prmt_size >= long(1)<<32 ) { printf("%s %d: Model size GEQ 4GB\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
    if ( inv_prmt_size <= 0           ) { printf("%s %d: Model size LEQ 0GB\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
    // NOTE: There may be issues when input files being read are larger than 4 GB.


    // ---- fwd

    Mx_soln = soln_M_sizes.at(0);  Nx_soln = Mx_soln + 1; 
    My_soln = soln_M_sizes.at(1);  Ny_soln = My_soln + 1; 


    Map_bdry_type[{'x','L'}] = bdry_type_L.at(0);
    Map_bdry_type[{'x','R'}] = bdry_type_R.at(0);

    Map_bdry_type[{'y','L'}] = bdry_type_L.at(1);
    Map_bdry_type[{'y','R'}] = bdry_type_R.at(1);

    // NOTE: We have assumed that there is only 1 MPI process for 1 src 
    //       and that all boundaries are of free surface type.
    for ( const char c_dir : {'x','y'} )
    for ( const char c_LR  : {'L','R'} )
    { assert ( ( Map_bdry_type.at( {c_dir,c_LR} ) == 'F' ) ); }
    // [2023/01/10]
    // NOTE: We need double (()), otherwise the preprocessor complains:
    //       "error: macro "assert" passed 2 arguments, but takes just 1"
    //       This is because it treated the , as separator of two tokens.
}


#endif