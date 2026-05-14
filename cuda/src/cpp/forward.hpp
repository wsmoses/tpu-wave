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
        // NOTE: We have made the name choices that 
        //           'record_rcv' means either synthetic or real 'data' collected at receiver locations
        //           'RESULT_rcv' means the simulated solution                    at receiver locations
        //       We could have used 'solution_rcv' in place of 'RESULT_rcv', which may be more expressive, 
        //       but 'solution_rcv' and 'record_rcv' have different word length, and they do tend to appear
        //       together. To better distinguish these two words, we capitalize 'RESULT'.
        // 
        //           Map_grid_record_rcv should be used for input ONLY and here ONLY (in this folder); 
        //       For output in ../misc/generate_data, Map_grid_RESULT_rcv should be used (in member functions
        //       forward_simulation and output_solution_record).
        // 
        //           Map_grid_RESULT_rcv is also used here to store the solution to compare against the records 
        //       stored in Map_grid_record_rcv for misfit calculation.
        

        double data_misfit;         // aggregated data misfit from all receivers corresponding to this source
        std::map< std::array<char, ns_forward::N_dir> , std::vector< double > > Map_grid_misfit_rcv;
        // NOTE: The above map from grid_type to a vector of doubleS (this vector should have the same length as 
        //       N_rcvs, i.e., the number of receivers on this grid) is used to record the data misfit for each 
        //       receiver corresponding to this source.
        //           We could just use one double (data_misfit) to accumulate the data misfit from all receivers. 
        //       Decided to use a complex struct such as Map_grid_misfit_rcv and have a record for each receiver 
        //       individually so that it may be more convenient later to investigate the impact of data coverage
        //       (in terms of setting up experiments). 


        // a function calculate_misfit
        //            aggregate_misfit  // do we really need this one ?

        // define another misfit in Inv_Specs ? aggregate // and a function aggregate_misfit (we can reset Inv_Specs.data_misfit there)


        // constructor
        // Class_Forward_Specs () {}

        void process_src_locations ( struct_src_input & src_input );

        void process_rcv_locations ( std::vector< struct_rcv_input > & Vec_Rcv_Input );

        
        void output_solution_record ( std::string folder_name );

        void output_energy ( std::string folder_name );


        // void forward_simulation ();   // this one correspond to all F (weak) boundary conditions

        template<int cpst_N = 0 , char cpst_S = '0'>
        void forward_simulation_periodic_y ();

        void input_solution_record ( std::string folder_name );
        
};

#endif
