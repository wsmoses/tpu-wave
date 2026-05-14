#include "input_processing.hpp"
#include "namespace_input.hpp"

#include <getopt.h>

using namespace ns_input;

//-----------------------------------------------//
//------------- Function defintiion -------------//
//-----------------------------------------------//
void file_input_processing ( std::string file_name )
{
    std::string input_line;
    std::ifstream input_file(file_name);
    if ( input_file.is_open() )
    {   
        while ( getline ( input_file , input_line ) )
        {
            // std::cout << input_line << std::endl;                // print the unprocessed line to screen

            std::string string_delimiter = "_";                     // use "_" as delimiter in input file to avoid ambiguity
            size_t bgn_pos, end_pos;

            // first 'content' goes to input_name, which is used to decide how the following numbers should be processed
            bgn_pos = 0;
            end_pos = input_line.find( string_delimiter , bgn_pos );                
            std::string input_name = input_line.substr( bgn_pos , end_pos - bgn_pos );

            if ( strcmp( input_name.c_str(), "model" ) == 0 )
            {
                // for ( int i=0; i<ns_forward::N_dir; i++ )
                for ( const char& c_dir : {'X','Y'} )
                {
                    int i_dir = ns_forward::XY_to_01(c_dir);

                    bgn_pos = end_pos + string_delimiter.length();
                    end_pos = input_line.find( string_delimiter , bgn_pos );
                    prmt_M_sizes.at(i_dir) = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );
                }
            }

            if ( strcmp( input_name.c_str(), "grids" ) == 0 )
            {
                // for ( int i=0; i<ns_forward::N_dir; i++ )
                for ( const char& c_dir : {'X','Y'} )
                {
                    int i_dir = ns_forward::XY_to_01(c_dir);

                    bgn_pos = end_pos + string_delimiter.length();
                    end_pos = input_line.find( string_delimiter , bgn_pos );
                    soln_M_sizes.at(i_dir) = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );
                }
            }

            if ( strcmp( input_name.c_str(), "dxM" ) == 0 )
            {
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                num_dx_prmt = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );

                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                den_dx_prmt = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );
            }

            if ( strcmp( input_name.c_str(), "dxS" ) == 0 )
            {
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                num_dx_soln = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );

                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                den_dx_soln = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );
            }

            if ( strcmp( input_name.c_str(), "bdry" ) == 0 ) 
            {
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                std::string B_type = input_line.substr( bgn_pos , end_pos - bgn_pos );

                { int i_dir = ns_forward::XY_to_01('X');  bdry_type_L.at(i_dir) = toupper( B_type[0] );  bdry_type_R.at(i_dir) = toupper( B_type[1] ); }
                { int i_dir = ns_forward::XY_to_01('Y');  bdry_type_L.at(i_dir) = toupper( B_type[2] );  bdry_type_R.at(i_dir) = toupper( B_type[3] ); }
            }

            if ( strcmp( input_name.c_str(), "Nt" ) == 0 ) 
            {
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                Nt = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );
            } 

            if ( strcmp( input_name.c_str(), "MAXdt" ) == 0 ) 
            {
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                dt_max = std::stod( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str() ); 
            } 

            if ( strcmp( input_name.c_str(), "CFL" ) == 0 ) 
            {
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                CFL_constant = std::stod( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str() ); 
            } 

            if ( strcmp( input_name.c_str(), "frequency" ) == 0 ) 
            {
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                central_f = std::stod( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str() ); 
            } 

            if ( strcmp( input_name.c_str(), "delay" ) == 0 ) 
            {
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                time_delay = std::stod( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str() ); 
            } 

            if ( strcmp( input_name.c_str(), "device" ) == 0 ) 
            {
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                device_number = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );
            } 

        }
        input_file.close();
    } // if ( input_file.is_open() )
    else 
    {
        printf( "Unable to open file: %s\n", file_name.c_str() ); fflush(stdout); exit(0); 
    }
    
} // file_input_processing()


