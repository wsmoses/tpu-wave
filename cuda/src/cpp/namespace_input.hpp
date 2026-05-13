#ifndef NAMESPACE_INPUT_H
#define NAMESPACE_INPUT_H

#include <array>
#include <math.h>
#include <assert.h>
#include <string>
#include <map>

#include "namespace_forward.hpp"
#include "container_run_time.hpp"

namespace namespace_input
{
    inline std::string medium_name = "homogeneous";

    inline std::array<char,ns_forward::N_dir> bdry_type_L {'F','F'};
    inline std::array<char,ns_forward::N_dir> bdry_type_R {'F','F'};

    inline std::map< std::pair<char,char> , char > Map_bdry_type;

    inline std::array<int, ns_forward::N_dir> prmt_M_sizes {-1,-1};
    inline std::array<int, ns_forward::N_dir> soln_M_sizes {-1,-1};

    inline long Mx_soln, Nx_soln;
    inline long My_soln, Ny_soln;

    inline long Mx_prmt, Nx_prmt;
    inline long My_prmt, Ny_prmt;

    inline long inv_prmt_stride_x = -(1<<30);
    inline long inv_prmt_stride_y = -(1<<30);

    inline long        inv_prmt_size;
    inline long PADDED_inv_prmt_size;

    inline int num_dx_prmt = -(1<<30);  
    inline int den_dx_prmt = 1;
    
    inline int num_dx_soln = -(1<<30);  
    inline int den_dx_soln = 1;

    inline int    Nt           =    -1;
    inline double dt_max       = 1<<30;
    inline double CFL_constant = 0.625;

    inline double central_f    = 5.;
    inline double time_delay   = 0.25;

    inline char c_energy       = 'Y';
    inline bool bool_energy    = true;

    inline bool bool_visual = true;

    inline double dx = (double) -(1<<30);
    inline double dt = (double) -(1<<30);

    inline double source_scaling = 1.;

    inline int  device_number = -1;

    inline run_time_vector<double> Record_E_k;
    inline run_time_vector<double> Record_E_p0;
    inline run_time_vector<double> Record_E_p1;
}
namespace ns_input = namespace_input;

inline void print_discretization_parameters ( double const min_velocity , double const max_velocity )
{
}

inline void ns_input_derived_variables ()
{
    using namespace ns_input;

    Mx_prmt = prmt_M_sizes.at(0);  Nx_prmt = Mx_prmt + 1;
    My_prmt = prmt_M_sizes.at(1);  Ny_prmt = My_prmt + 1;

    inv_prmt_stride_x = Ny_prmt;
    inv_prmt_stride_y = 1;

           inv_prmt_size =   Nx_prmt *         Ny_prmt;
    PADDED_inv_prmt_size = ( Nx_prmt + 1 ) * ( Ny_prmt + 1 );

    if ( inv_prmt_size >= long(1)<<32 ) { printf("%s %d: Model size GEQ 4GB\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
    if ( inv_prmt_size <= 0           ) { printf("%s %d: Model size LEQ 0GB\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

    Mx_soln = soln_M_sizes.at(0);  Nx_soln = Mx_soln + 1; 
    My_soln = soln_M_sizes.at(1);  Ny_soln = My_soln + 1; 

    Map_bdry_type[{'x','L'}] = bdry_type_L.at(0);
    Map_bdry_type[{'x','R'}] = bdry_type_R.at(0);

    Map_bdry_type[{'y','L'}] = bdry_type_L.at(1);
    Map_bdry_type[{'y','R'}] = bdry_type_R.at(1);

    for ( const char c_dir : {'x','y'} )
    for ( const char c_LR  : {'L','R'} )
    { assert ( ( Map_bdry_type.at( {c_dir,c_LR} ) == 'F' ) ); }
}

#endif