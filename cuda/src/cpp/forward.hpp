#ifndef FORWARD_H
#define FORWARD_H

#include "src_rcv.hpp"
#include "namespace_input.hpp"

#include <algorithm>

class Class_Forward_Specs 
{
    public:

        // ---- Map from grid types to pointers of Class_Grid
        std::map< std::array<char, ns_forward::N_dir> , Class_Grid * > Map_Grid_pointers {};

        struct_src_forward src_forward;

        std::map< std::array<char, ns_forward::N_dir> , int >                                    Map_grid_N_rcvs;
        std::map< std::array<char, ns_forward::N_dir> , std::vector< struct_rcv_forward > >      Map_grid_struct_rcv_forward;

        std::map< std::array<char, ns_forward::N_dir> , std::vector< run_time_matrix<ns_type::host_precision> > > Map_grid_record_rcv;  // involve memory
        std::map< std::array<char, ns_forward::N_dir> , std::vector< run_time_matrix<ns_type::host_precision> > > Map_grid_RESULT_rcv;  // involve memory
        
	double data_misfit;         // aggregated data misfit from all receivers corresponding to this source
};

#endif
