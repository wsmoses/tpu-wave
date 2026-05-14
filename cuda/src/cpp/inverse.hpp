#ifndef INVERSE_H
#define INVERSE_H

#include "src_rcv.hpp"
#include "namespace_input.hpp"

class Class_Inverse_Specs 
{
    public:

        std::map< std::string , run_time_vector< double > > Map_inv_prmt;
        // [2023/06/09]
        // NOTE: I think we can use double above.

        std::vector< struct_src_input > Vec_Src_Input;
        std::map< int , std::vector< struct_rcv_input > > Map_Vec_Rcv_Input;

        // NOTE: We have removed the following structs that involve double MapS
        //           std::map< int , std::map< std::array<char, N_dir> , int > >                                    Map_Map_grid_N_rcvs;
        //           std::map< int , std::map< std::array<char, N_dir> , std::vector< struct_rcv_forward > > >      Map_Map_grid_struct_rcv_forward;
        //           std::map< int , std::map< std::array<char, N_dir> , std::vector< run_time_matrix<double> > > > Map_Map_grid_record_rcv;
        //       on the premise that the processing of receivers should take place
        //       on the 'simulation' processes rather than on the 'master' process. 
        //           In particular, the allocation of memory for solution records 
        //       should take place on 'simulation process'. 
        //           Moreover, during the inversion, it's not the recorded solution
        //       that needs to be sent over, but rather the processed information 
        //       (e.g., the calculated misfit).
        //
        //           However, some preliminary processing of source information may be placed 
        //       back here again. We may add the following struct Map_Src_Forward as a member
        //       variable of this class:
        //           std::map< int , struct_src_forward > Map_Src_Forward;
 
        double data_misfit;         // aggregated data misfit for all sources
 
        // constructor
        // Class_Inverse_Specs () {}
 
        void input_inverse_parameter ( std::string file_name, std::string prmt_name, ns_input::InputParams &params );

        void input_SrcRcv_locations ( std::string file_name ); 

        void aggregate_data_misfit ( double fwd_data_misfit ) { this->data_misfit += fwd_data_misfit; }

        // NOTE: Decided to implement the input of solution records as a member function of Fwd_Specs, 
        //       i.e., input_solution_record ();
        //           Moreover, decided to organize the solution records as separated files, indexed by
        //       the sources, and are already present on the local storage space that each simulation
        //       process has access to.
};

#endif