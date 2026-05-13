#ifndef CUDA_CONTAINER_RUN_TIME_CUH
#define CUDA_CONTAINER_RUN_TIME_CUH

#include <iostream>
#include <fstream>
#include <string.h>
#include <assert.h>

template<typename T> class cuda_run_time_vector_shallow_copy;
template<typename T> class cuda_run_time_matrix_shallow_copy;


template<typename T>
class cuda_run_time_vector
{
    public:
        long length = 0;
        T * ptr = nullptr;


        std::string name = "undefined";


        cuda_run_time_vector(){}  // Empty constructor in case we need to declare a cuda_run_time_vector before knowing its 
                                  // size.
                                  //
                                  // NOTE: Though with an empty body, it is still an user-defined default constructor. 
                                  //       Sometimes, its implicit generation may be inhibited for various reasons.
                                  //       ( reference: https://en.cppreference.com/w/cpp/language/default_constructor )

        void allocate_memory (long const L) // member function
        {
            if ( ptr != nullptr ) 
            { 
                printf( "%s %d: ptr is already allocated for %s.\n", 
                        __FILE__, __LINE__, this->name.c_str() ); fflush(stdout); exit(0); 
            }

            if (L <= 0) 
            { 
                printf( "%s %d: L <= 0 for %s.\n", 
                        __FILE__, __LINE__, this->name.c_str() ); fflush(stdout); exit(0); 
            }

            cudaError_t malloc_result = cudaMalloc( &ptr, L*sizeof(T) );
            if ( malloc_result != cudaSuccess ) 
            { 
                printf( "(%s) for %s %d %s.\n", cudaGetErrorString(malloc_result),
                        __FILE__, __LINE__, this->name.c_str() ); fflush(stdout); exit(0); 
            }
            cudaMemset ( ptr, 0, L*sizeof(T) );    // Memory is not cleared with cudaMalloc;
            // NOTE: Alignment should have been accounted for by cudaMalloc.


            this->length  =  L;
            // NOTE: Assignment of L to this->length is here because sometimes we push_back an empty 
            //       run_time_vector to std::vector and call allocate_memory explicitly afterwards.
            //           In this case, it would be 'easy to forget' setting this->length accordingly
            //       if we don't include it here.
        }


        cuda_run_time_vector (long const L) // a user-supplied 'non-speical' ctor
            { allocate_memory (L); }
            
        // move ctor
        cuda_run_time_vector( cuda_run_time_vector<T> && v ) noexcept
        {
            this->length = v.length;  this->ptr = v.ptr;

            v.length = 0;             v.ptr = nullptr;
        }
        // NOTE: I think the above implementation is 'safe'; move ctor does not call allocate_memory (),
        //       therefore we can directly assign to this->ptr and set v.ptr to nullptr. 

        // move assignment operator
        cuda_run_time_vector<T> & operator=( cuda_run_time_vector<T> && v ) = delete;
        // NOTE: For rationale behind '= delete', see the comments in folder rule_035.


        // dtor
        ~cuda_run_time_vector() 
        { 
            cudaError_t free_result = cudaFree(ptr); 
            if ( free_result != cudaSuccess ) 
                { printf("cudaFree unsuccessful: %s.\n", cudaGetErrorString(free_result)); exit(0); }
        }
        // Questions: Is cudaFree(ptr) enough? What might be the corner cases that I haven't thought of?

}; 

#endif
