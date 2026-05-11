#include "namespace_input.hpp"

namespace namespace_input
{
    std::string medium_name = "homogeneous";

    std::array<char,ns_forward::N_dir> bdry_type_L {'F','F'};
    std::array<char,ns_forward::N_dir> bdry_type_R {'F','F'};

    std::map< std::pair<char,char> , char > Map_bdry_type;

    std::array<int, ns_forward::N_dir> prmt_M_sizes {-1,-1};
    std::array<int, ns_forward::N_dir> soln_M_sizes {-1,-1};

    long Mx_soln, Nx_soln;
    long My_soln, Ny_soln;

    long Mx_prmt, Nx_prmt;
    long My_prmt, Ny_prmt;

    long inv_prmt_stride_x = -(1<<30);
    long inv_prmt_stride_y = -(1<<30);

    long        inv_prmt_size;
    long PADDED_inv_prmt_size;

    int num_dx_prmt = -(1<<30);  
    int den_dx_prmt = 1;
    
    int num_dx_soln = -(1<<30);  
    int den_dx_soln = 1;

    int    Nt           =    -1;
    double dt_max       = 1<<30;
    double CFL_constant = 0.625;

    double central_f    = 5.;
    double time_delay   = 0.25;

    char c_energy       = 'Y';
    bool bool_energy    = true;

    bool bool_visual = true;

    double dx = (double) -(1<<30);
    double dt = (double) -(1<<30);

    double source_scaling = 1.;

    int  device_number = -1;

    run_time_vector<double> Record_E_k ;
    run_time_vector<double> Record_E_p0;
    run_time_vector<double> Record_E_p1;
}
