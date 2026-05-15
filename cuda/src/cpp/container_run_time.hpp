#ifndef CONTAINER_RUN_TIME_H
#define CONTAINER_RUN_TIME_H

#include <iostream>
// #include <fstream>

#include <math.h>
#include <string.h>

#include "container_compile_time.hpp"

template<typename T> class run_time_vector;
template<typename T> class run_time_matrix;


template<typename T>
class run_time_vector
{
    public:
        int length = -1<<30;
        T * ptr = nullptr;

        // NOTE: We only need/want the ptr to be aligned, not the class itself.
        //       There can be a difference if other members, e.g., length in 
        //       this case, is placed before ptr.
        //           If the class is desired to be aligned in a specific way, 
        //       one can use the alignas() specifier.

        run_time_vector(){}  // Empty constructor in case we need to declare a run_time_vector before knowing its 
                             // size.
                             //
                             // NOTE: Though with an empty body, it is still an user-defined default constructor. 
                             //       Sometimes, its implicit generation may be inhibited for various reasons.
                             //       ( reference: https://en.cppreference.com/w/cpp/language/default_constructor )

        void allocate_memory (int const L) // member function
        {
            this->length  =  L;
            // NOTE: Assignment of L to this->length is here because sometimes we push_back an empty 
            //       run_time_vector to std::vector and call allocate_memory explicitly afterwards.
            //           In this case, it would be 'easy to forget' setting this->length accordingly
            //       if we don't include it here.

// do we want to make alignment dependent on T? 
// we should probably take alignment out, maybe put it into the same namespace where prcs_type is defined
            constexpr int alignment = 32;    // 32 bytes; 
            int byte_size = ( ( L*sizeof(T) + alignment - 1 ) / alignment ) * alignment;

            ptr = static_cast<T *>( aligned_alloc(alignment, byte_size) );
        }
        // NOTE: If empty constructor (currently the default constructor) is called, the above
        //       function needs to be called explicitly to allocate memory.
        //
        // NOTE: Do we want to make 'alignment' in the above member function a global variable 
        //       (possibly defined in some other file) or not? 
        //           I think leave it here is better - locality, hence more expressive.


        run_time_vector (int const L) // a user-supplied 'non-speical' constructor
            { allocate_memory (L); }
            


        T& at (int const i) // read and write access
        {
            if ( ptr == nullptr   ) { printf("Error: nullptr - at - run_time_vector.\n"); fflush(stdout); }
            if ( i<0 || i>=length ) { printf("Out of bound access - at - run_time_vector. i=%d, length=%d\n", i, length);   
                                      printf("    Possible cue 1): Forgot to change the input parameters in compile_time_namespace.hpp?\n");
                                      printf("    Possible cue 2): Forgot to change the medium name/path in run time command?\n"); 
				      fflush(stdout); exit(0); }

            return ptr[i];
        }

}; 


#endif
