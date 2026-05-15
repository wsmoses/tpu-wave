#include "inverse.hpp"
#include "namespace_input.hpp"

#include <set>

void Class_Inverse_Specs::input_inverse_parameter ( std::string file_name, std::string prmt_name, ns_input::InputParams &params )
{
    using namespace ns_input;
    using namespace std;
    using prmt_data_type = double;

    // ---- check if the datum length is as expected (e.g., expecting floats that take 4 bytes each)
    // if ( sizeof(float) != 4 ) { printf("\nFloat does not take 4 bytes!\n"); fflush(stdout); exit(0); }
    // ---- test the endianness of the system
    // int test_Endian = 1; if (*(char *)&test_Endian == 1) printf("\nLittle-Endian\n\n"); else printf("\nBig-Endian\n\n");


    if ( strcmp( prmt_name.c_str(), "rho" ) != 0 
      && strcmp( prmt_name.c_str(), "vp"  ) != 0 
      && strcmp( prmt_name.c_str(), "vs"  ) != 0 )
    {
        printf("Parameter name not expected.\n"); fflush(stdout); exit(0);
        // NOTE: we may input/interface using other physical parameters.
    }


    // variables related to reading the input file
    streampos byte_size;                                                // count for the size of data read in (with unit byte)
    ifstream  data_file;                                                // file class

    char           * memblock = nullptr;                                // memory block to hold the uninterpretted data
    prmt_data_type * data_ptr = nullptr;                                // pointer used to interpret the memory block
    

    // read in raw data (in bytes)
    data_file.open(file_name, ios::in | ios::binary | ios::ate);        // open data, place the pointer at the end
    if ( data_file.is_open() ) 
    {
        byte_size = data_file.tellg();                                  // count the size of the raw data in byte
        memblock = new char [byte_size];                                // allocate memblock
        data_file.seekg (0, ios::beg);                                  // place the pointer to the beginning
        data_file.read (memblock, byte_size);                           // read in the data
        data_file.close();                                              // close data
    }
    else 
        { printf( "Unable to open file: %s\n", file_name.c_str() ); fflush(stdout); exit(0); }
    
    // check the size
    if ( static_cast<long>( params.inv_prmt_size * sizeof( prmt_data_type ) ) != static_cast<long>( byte_size ) ) 
    {   
        printf( "Size of the input data file %s not as expected : %ld (model size) vs %ld (byte size).\n", 
                                             file_name.c_str(), params.inv_prmt_size, static_cast<long>( byte_size ) );
        fflush(stdout); exit(0); 
    }


    // interpret and copy the data to inv prmts
    data_ptr = (prmt_data_type *) memblock;


    if ( Map_inv_prmt.count( prmt_name ) > 0 )
        { printf("The input parameter %s has already been allocated memory space.", prmt_name.c_str()); fflush(stdout); exit(0); }
    Map_inv_prmt[ prmt_name ].allocate_memory ( params.PADDED_inv_prmt_size );

    run_time_vector<double> * model_data = nullptr;
    model_data = & ( Map_inv_prmt.at( prmt_name ) );
    // if ( model_data == nullptr ) 
    //     { printf("Pointer model_data remains nullptr %s %d.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
    

    // NOTE: The inverse parameters stored in Inv_Specs have been padded with an extra number 
    //       (which should remain zero) on each direction so that the interpolation in grid
    //       does not step out of bound and cause the 'ghostly' bug.
    //           PAY SPECIAL ATTENTION to the size and stride of the inverse parameters 
    //       acutally stored in Inv_Specs : THE SIZE IS 
    //              Inv_Specs.PADDED_inv_prmt_size = ( Nx_prmt + 1 ) * ( Ny_prmt + 1 )
    //           THE STRIDE IS 
    //              ( Ny_prmt + 1 ).
    const int PADDED_stride = params.Ny_prmt + 1;
     
    int data_count = 0;
    for ( int ix = 0; ix < params.Nx_prmt; ix++ )
    {
        for ( int iy = 0; iy < params.Ny_prmt; iy++ )
        {        
            int             i_prmt   = ix * PADDED_stride + iy;    // PADDED_stride = Ny_prmt + 1
            model_data->at( i_prmt ) = data_ptr[data_count];
            data_count++;
        }
    }
    // NOTE: In the above assignment, implicit cast from may take place.
    // 
    // NOTE: The above assignment assumes data on both sides are ordered consistently;
    //       In this case, x should be the outer direction, y should be the inner direction.
    

    data_ptr = nullptr;
    delete[] memblock;


    // check the compatibility of parameters if under periodic boundary condition 
    //     NOTE: NOT currently useful since all boundaries are assumed to be free surface.
    // ---- x direction
    if ( params.bdry_type_L.at(0) == 'P' && params.bdry_type_R.at(0) == 'P' )
    {
        printf("\n ---- Periodic parameter check took place on x direction. ---- \n");
        for ( int iy = 0; iy < params.Ny_prmt; iy++ )
        {
            int iL =             0   * PADDED_stride + iy;    // PADDED_stride = Ny_prmt + 1
            int iR = ( params.Nx_prmt - 1 ) * PADDED_stride + iy;    // PADDED_stride = Ny_prmt + 1

            if ( model_data->at(iL) != model_data->at(iR) )
                { printf("Input inverse parameter not compatible with periodic boundary condition %s %d.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
        }
    }
    // ---- y direction
    if ( params.bdry_type_L.at(1) == 'P' && params.bdry_type_R.at(1) == 'P' )
    {
        printf("\n ---- Periodic parameter check took place on y direction. ---- \n");
        for ( int ix = 0; ix < params.Nx_prmt; ix++ )
        {
            int iL = ix * PADDED_stride +             0  ;    // PADDED_stride = Ny_prmt + 1
            int iR = ix * PADDED_stride + ( params.Ny_prmt - 1 );    // PADDED_stride = Ny_prmt + 1

            if ( model_data->at(iL) != model_data->at(iR) )
                { printf("Input inverse parameter not compatible with periodic boundary condition %s %d.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
        }
    }

} // input_inverse_parameter ()


void Class_Inverse_Specs::input_SrcRcv_locations ( std::string file_name )
{
    constexpr int N_dir = ns_forward::N_dir;

    if ( N_dir != 2 ) { printf("Only intended to work in 2D.\n"); fflush(stdout); exit(0); }

    // NOTE: By applying XY_to_01 () below when processing grid_type and V_Loc, the input src 
    //       and rcv information, which are expressed for 'X' and 'Y', should be converted to 
    //       'x' (0) and 'y' (1).

    int src_count = 0;
    int rcv_count = 0;

    std::string input_line;
    std::ifstream input_file(file_name);
    if ( input_file.is_open() )
    {
        while ( getline ( input_file , input_line ) )
        {
            // use "_" as delimiter in input file to avoid ambiguity
            std::string string_delimiter = "_";
            size_t bgn_pos, end_pos;

            // first 'content' goes to input_name, which is used to decide how the following numbers should be processed
            bgn_pos = 0;
            end_pos = input_line.find( string_delimiter , bgn_pos );
            std::string input_name = input_line.substr( bgn_pos , end_pos - bgn_pos );


            // If a matching line for source is found
            if ( strcasecmp( input_name.c_str(), "Src" ) == 0 )
            {
                printf( "    Processing %s ---- %d\n" , input_name.c_str() , src_count );

                // second 'content' uniquely identifies this source -> src_index
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                int src_index = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );

                // third 'content' indicates the grid this source is on
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                std::string input_string = input_line.substr( bgn_pos , end_pos - bgn_pos );
                std::array<char, N_dir> grid_type {};  // for ( int i = 0; i < N_dir; i++ ) { grid_type[i] = input_string.at(i); }
                { 
                    int count = 0; 
                    for ( const char& c_dir : {'X','Y'} ) { grid_type.at( ns_forward::XY_to_01(c_dir) ) = input_string.at(count); count++; }
                }


                // the next 4 'contents' indicate the location of this source (inputted as rationals)
                std::array< int, N_dir * 2 > V_Loc {}; 
                // for ( int i = 0; i < N_dir * 2; i++ )
                // {
                //     bgn_pos = end_pos + string_delimiter.length();
                //     end_pos = input_line.find( string_delimiter , bgn_pos );
                //     V_Loc[i]  = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );
                // }

                for ( const char& c_dir : {'X','Y'} ) 
                {
                    int i_dir = ns_forward::XY_to_01( c_dir );
                    for ( int i = 0; i < 2; i++ ) // num and den
                    {
                        bgn_pos = end_pos + string_delimiter.length();
                        end_pos = input_line.find( string_delimiter , bgn_pos );

                        int i_loc = i_dir * 2 + i;
                        V_Loc[i_loc]  = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );
                    }
                }

                Vec_Src_Input.push_back ( struct_src_input { src_index , grid_type , V_Loc } );
                Map_Vec_Rcv_Input[src_index] = {};  // src_index may not be consecutive; also, it may not START with 0

                src_count += 1;                
            }   


            // If a matching line for receiver is found
            if ( strcasecmp( input_name.c_str(), "Rcv" ) == 0 )
            {
                printf( "    Processing %s ---- %d\n" , input_name.c_str() , rcv_count );

                // second 'content' identifies which source this receiver corresponds to -> src_index
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                int src_index = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );

                // if ( !Map_Vec_Rcv_Input.contains(src_index) )    // contains() is introduced in -20
                if ( Map_Vec_Rcv_Input.count(src_index) == 0 )
                { 
                    printf( "This receiver (with src_index %d) does not have a corresponding source, yet\n", src_index ); 
                    fflush(stdout); exit(0); 
                }
                // NOTE: for the above check to work, source with src_index needs to be 
                //       placed above receivers with src_index in the input file SrcRcv.txt

                // third 'content' identifies the receiver within this source -> rcv_index
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                int rcv_index = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );

                // fourth 'content' indicates the grid this receiver is on
                bgn_pos = end_pos + string_delimiter.length();
                end_pos = input_line.find( string_delimiter , bgn_pos );
                std::string input_string = input_line.substr( bgn_pos , end_pos - bgn_pos );
                std::array<char, N_dir> grid_type {};  // for ( int i = 0; i < N_dir; i++ ) { grid_type[i] = input_string.at(i); } 
                { 
                    int count = 0; 
                    for ( const char& c_dir : {'X','Y'} ) { grid_type.at( ns_forward::XY_to_01(c_dir) ) = input_string.at(count); count++; }
                }

                // the next N_dir * 2 'contents' indicate the location of this source (inputted as rationals)
                std::array< int, N_dir * 2 > V_Loc {}; 
                // for ( int i = 0; i < N_dir * 2; i++ )
                // {
                //     bgn_pos = end_pos + string_delimiter.length();
                //     end_pos = input_line.find( string_delimiter , bgn_pos );
                //     V_Loc[i]  = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );
                // }

                for ( const char& c_dir : {'X','Y'} ) 
                {
                    int i_dir = ns_forward::XY_to_01( c_dir );
                    for ( int i = 0; i < 2; i++ ) // num and den
                    {
                        bgn_pos = end_pos + string_delimiter.length();
                        end_pos = input_line.find( string_delimiter , bgn_pos );

                        int i_loc = i_dir * 2 + i;
                        V_Loc[i_loc]  = strtol( input_line.substr( bgn_pos , end_pos - bgn_pos ).c_str(), nullptr, 10 );
                    }
                }

                Map_Vec_Rcv_Input.at(src_index).push_back ( struct_rcv_input { src_index , rcv_index , grid_type , V_Loc } );

                rcv_count += 1;
            }
        }
        input_file.close();
    }
    else 
        { printf( "Unable to open file: %s\n", file_name.c_str() ); fflush(stdout); exit(0); }


    // ---- check the LENGTH for Src
    printf( "Total lines of sources   processed:  %4d\n" , src_count );
    if ( static_cast<unsigned long> ( src_count ) != Vec_Src_Input.size() ) 
        { printf( "Number of sources   processed do not agree: %d %ld\n", src_count, Vec_Src_Input.size() ); fflush(stdout); exit(0); }


    // ---- check the LENGTH for Rcv
    printf( "Total lines of receivers processed:  %4d\n" , rcv_count );
    int rcv_count_map = 0; for ( const auto& it_map : Map_Vec_Rcv_Input ) { rcv_count_map += it_map.second.size(); }
    if ( rcv_count != rcv_count_map ) 
        { printf( "Number of receivers processed do not agree: %d %d\n", rcv_count, rcv_count_map ); fflush(stdout); exit(0); }


    // ---- check whether there is duplication in SOURCE LOCATIONS
    printf( "    Checking if there is duplication in source locations.\n" );
    std::set< std::pair< std::array<char, N_dir> , std::array< int, N_dir * 2 > > > Set_Src; 
    for ( const auto & it_vec : Vec_Src_Input )
    {
        auto insert_result = Set_Src.insert( std::make_pair( it_vec.src_grid_type , it_vec.src_location ) );
        // NOTE: to have a well-defined set, the type of the element needs to have a well-defined comparison operator.
        if ( !insert_result.second )
        {
            printf( "Source with index %d not inserted.\n", it_vec.src_index );
            printf( "Possible duplicate location ---- abort.\n" );  fflush(stdout); exit(0);
        }
    }

    // ---- check whether there is duplication in SOURCE INDICES
    printf( "    Checking if there is duplication in source indices.\n" );
    std::set< int > Set_Src_Index; 
    for ( const auto & it_vec : Vec_Src_Input )
    {
        auto insert_result = Set_Src_Index.insert( it_vec.src_index );
        if ( !insert_result.second )
        {
            printf( "Source with index %d not inserted.\n", it_vec.src_index );
            printf( "Possible duplicate index ---- abort.\n" );  fflush(stdout); exit(0);
        }
    }

    
    for ( const auto & it_map : Map_Vec_Rcv_Input )
    {
        // For each src_index, check if there is duplication in RECEIVER LOCATIONS
        printf( "    Checking if there is duplication in receiver locations for source %d.\n", it_map.first );
        std::set< std::pair< std::array<char, N_dir> , std::array< int, N_dir * 2 > > > Set_Rcv; 
        for ( const auto & it_vec : it_map.second )
        {
            auto insert_result = Set_Rcv.insert( std::make_pair( it_vec.rcv_grid_type , it_vec.rcv_location ) );
            if ( !insert_result.second )
            {
                printf( "Source %d - Receiver %d not inserted.", it_vec.src_index, it_vec.rcv_index );
                printf( "\n Possible duplicate location ---- abort.\n" );  fflush(stdout); exit(0);
            }
        }

        // For each src_index, check if there is duplication in RECEIVER INDICES
        printf( "    Checking if there is duplication in receiver index for source %d.\n", it_map.first );
        std::set< int > Set_Rcv_Index; 
        for ( const auto & it_vec : it_map.second )
        {
            auto insert_result = Set_Rcv_Index.insert( it_vec.rcv_index );
            if ( !insert_result.second )
            {
                printf( "Source %d - Receiver %d not inserted.", it_vec.src_index, it_vec.rcv_index );
                printf( "\n Possible duplicate index ---- abort.\n" );  fflush(stdout); exit(0);
            }
        }
    }
} 