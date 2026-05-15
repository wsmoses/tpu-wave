#ifndef NAMESPACE_INPUT_H
#define NAMESPACE_INPUT_H

#include <array>
#include <math.h>
#include <assert.h>

#include "namespace_forward.hpp"
#include "container_run_time.hpp"

namespace namespace_input
{
    struct InputParams {
        std::array<char,ns_forward::N_dir> bdry_type_L;
        std::array<char,ns_forward::N_dir> bdry_type_R;

        std::array<int, ns_forward::N_dir> prmt_M_sizes;
        std::array<int, ns_forward::N_dir> soln_M_sizes;

        long Mx_soln = 0, Nx_soln = 0;
        long My_soln = 0, Ny_soln = 0;

        long Mx_prmt = 0, Nx_prmt = 0;
        long My_prmt = 0, Ny_prmt = 0;

        long inv_prmt_stride_x = 0;
        long inv_prmt_stride_y = 0;

        long        inv_prmt_size = 0;
        long PADDED_inv_prmt_size = 0;

        int num_dx_prmt = 0;
        int den_dx_prmt = 0;
        
        int num_dx_soln = 0;
        int den_dx_soln = 0;

        int    Nt = 0;
        double dt_max = 0.0;
        double CFL_constant = 0.0;

        double central_f = 0.0;
        double time_delay = 0.0;

        double dx = 0.0;
        double dt = 0.0;

        int  device_number = 0;
    };


inline void print_discretization_parameters ( double const min_velocity , double const max_velocity, InputParams &params )
{
    using ns_forward::N_dir;

    printf("\n");
    printf(" dx: %10.7f", params.dx);                            printf("%*c    ", 2, ' ');

    // printf(" total length (x): %10.7f", params.dx * params.Mx_soln);    printf("%*c    ", 2, ' ');
    // printf(" total length (y): %10.7f", params.dx * params.My_soln);    printf("%*c    ", 2, ' ');

    printf(" total length (x): %10.7f", params.dx * params.soln_M_sizes.at(0));    printf("%*c    ", 2, ' ');
    printf(" total length (y): %10.7f", params.dx * params.soln_M_sizes.at(1));    printf("%*c    ", 2, ' ');


    // [2023/01/15]
    // NOTE: We probably want to directly specify Mx_soln, My_soln, Mx_prmt, My_prmt
    //       and remove prmt_M_sizes and soln_M_sizes.

    
    printf(" dt: %10.7f", params.dt);                            printf("%*c    ", 2, ' ');
    printf(" total time: %10.7f", params.dt*params.Nt);                 printf("%*c  \n", 2, ' ');
    

    double min_lambda      = min_velocity / ( params.central_f * 2.5 );

    double effective_N_ppw = min_lambda / params.dx;
    printf("effective N_ppw:        %10.7f (min_velocity: %10.7f)\n", 
            effective_N_ppw,                min_velocity); 

    double effective_CFL_constant = ( params.dt / params.dx ) * ( sqrt( (double) N_dir ) * max_velocity );
    printf("effective CFL_constant: %10.7f (max_velocity: %10.7f)\n", 
            effective_CFL_constant,         max_velocity); 

    fflush(stdout);
}


inline void ns_input_derived_variables (InputParams &params)
{
    // ---- inv

    params.Mx_prmt = params.prmt_M_sizes.at(0);  params.Nx_prmt = params.Mx_prmt + 1;
    params.My_prmt = params.prmt_M_sizes.at(1);  params.Ny_prmt = params.My_prmt + 1;

    params.inv_prmt_stride_x = params.Ny_prmt;
    params.inv_prmt_stride_y = 1;

           params.inv_prmt_size =   params.Nx_prmt *         params.Ny_prmt;
    params.PADDED_inv_prmt_size = ( params.Nx_prmt + 1 ) * ( params.Ny_prmt + 1 );

    if ( params.inv_prmt_size >= long(1)<<32 ) { printf("%s %d: Model size GEQ 4GB\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
    if ( params.inv_prmt_size <= 0           ) { printf("%s %d: Model size LEQ 0GB\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
    // NOTE: There may be issues when input files being read are larger than 4 GB.


    // ---- fwd

    params.Mx_soln = params.soln_M_sizes.at(0);  params.Nx_soln = params.Mx_soln + 1; 
    params.My_soln = params.soln_M_sizes.at(1);  params.Ny_soln = params.My_soln + 1; 




}
}
namespace ns_input = namespace_input;


#endif