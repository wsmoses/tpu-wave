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
    private:
        bool bool_manage_memory = false;

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

            // ---- floating point is used below to avoid integer overflow
            if ( (double) ( L * sizeof(T) ) >= 1024.*1024.*1024. * 2. ) 
            { 
                printf( "%s %d: requested size for %s is larger than 2GB; if this "
                        "is intended, change the limit in the above if statement.\n", 
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


            bool_manage_memory = true;

            this->length  =  L;
            // NOTE: Assignment of L to this->length is here because sometimes we push_back an empty 
            //       run_time_vector to std::vector and call allocate_memory explicitly afterwards.
            //           In this case, it would be 'easy to forget' setting this->length accordingly
            //       if we don't include it here.
        }
        // NOTE: If empty constructor (currently the default constructor) is called, the above
        //       function needs to be called explicitly to allocate memory.


        bool return_bool_manage_memory () const { return bool_manage_memory; }


        cuda_run_time_vector (long const L) // a user-supplied 'non-speical' ctor
            { allocate_memory (L); }
            

        void copy ( cuda_run_time_vector<T> const & v )
        {
            // NOTE: Syntactically, this copy member function should "work" for self assignment. Its
            //       behavior is (arguably) "logically" correct. However, such "self" assignment is 
            //       an indication that some error or misuse may have occurred from the "application"
            //       side. We add this test (test_self_copy) to guard such misuse.
            if ( this == &v || this->ptr == v.ptr ) 
                { printf("%s %d Error: self copy.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }
            
            if ( this->ptr == nullptr || v.ptr == nullptr ) 
                { printf("%s %d Error:   nullptr.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            if ( length <= 0 || length != v.length ) 
                { printf("%s %d Error:    length.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            cudaError_t memcpy_result = cudaMemcpy ( ptr , v.ptr , v.length * sizeof(T) , cudaMemcpyDeviceToDevice );
            if ( memcpy_result != cudaSuccess ) 
            { 
                printf( "(%s) %s %d cudaMemcpy unsuccessful for %s.\n", cudaGetErrorString(memcpy_result),
                        __FILE__, __LINE__, this->name.c_str() ); fflush(stdout); exit(0); 
            }
        }
        // [2023/07/11]
        // NOTE: In "copy ()", we copy from "device" to "device", which is the difference from 
        //       "_copy_from_host ()" and "_copy_to_host ()".


        void mem_copy_from_host ( run_time_vector<T> const & v )
        {
            if ( this->ptr == nullptr || v.ptr == nullptr ) 
                { printf("%s %d Error:   nullptr.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            if ( length <= 0 || length != v.length ) 
                { printf("%s %d Error:    length.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            cudaError_t memcpy_result = cudaMemcpy ( ptr , v.ptr , v.length * sizeof(T) , cudaMemcpyHostToDevice );
            if ( memcpy_result != cudaSuccess ) 
            { 
                printf( "(%s) %s %d cudaMemcpy unsuccessful for %s.\n", cudaGetErrorString(memcpy_result),
                        __FILE__, __LINE__, this->name.c_str() ); fflush(stdout); exit(0); 
            }
        }


        template<typename host_precision>  // should be called host_T to avoid confusion
        void val_copy_from_host ( run_time_vector<host_precision> const & v )
        {
            if ( this->ptr == nullptr || v.ptr == nullptr ) 
                { printf("%s %d Error:   nullptr.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            if ( length <= 0 || length != v.length ) 
                { printf("%s %d Error:    length.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            if constexpr ( std::is_same_v< host_precision , T > )
            {
                printf("\n\n%s %d CONSIDER using mem_copy_from_host directly.\n\n", __FILE__, __LINE__);
                mem_copy_from_host ( v );
            }
            else 
            {
                run_time_vector<T> V_T (v.length);
                for ( long i=0; i<length; i++ ) { V_T.ptr[i] = static_cast <T> ( v.ptr[i] ); }

                mem_copy_from_host ( V_T );
            }
        }


        template<typename host_precision>
        void copy_from_host ( run_time_vector<host_precision> const & v )
        {
            if constexpr ( std::is_same_v< host_precision , T > )
                { mem_copy_from_host ( v ); }
            else
                { val_copy_from_host <host_precision> ( v ); }
        }
        // [2023/07/11]
        // NOTE: If calling copy_from_host (), the "if constexpr" branch in val_copy_from_host ()
        //       will never take place. Nevertheless, let's keep it there in case we want to call 
        //       val_copy_from_host () directly.


        // NOTE: In _copy_to_host, the src and dst pointers switch from the _copy_from_host.
        void mem_copy_to_host ( run_time_vector<T> & v ) const
        {
            if ( this->ptr == nullptr || v.ptr == nullptr ) 
                { printf("%s %d Error:   nullptr.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            if ( length <= 0 || length != v.length ) 
                { printf("%s %d Error:    length.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            cudaError_t memcpy_result = cudaMemcpy ( v.ptr , ptr , v.length * sizeof(T) , cudaMemcpyDeviceToHost );
            if ( memcpy_result != cudaSuccess ) 
            { 
                printf( "(%s) %s %d cudaMemcpy unsuccessful for %s.\n", cudaGetErrorString(memcpy_result),
                        __FILE__, __LINE__, this->name.c_str() ); fflush(stdout); exit(0); 
            }
        }


        template<typename host_precision>
        void val_copy_to_host ( run_time_vector<host_precision> & v ) const
        {
            if ( this->ptr == nullptr || v.ptr == nullptr ) 
                { printf("%s %d Error:   nullptr.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            if ( length <= 0 || length != v.length ) 
                { printf("%s %d Error:    length.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            if constexpr ( std::is_same_v< host_precision , T > )
            {
                printf("\n\n%s %d consider using mem_copy_to_host directly.\n\n", __FILE__, __LINE__);
                mem_copy_to_host ( v );
            }
            else 
            {
                run_time_vector<T> V_T (v.length);
                mem_copy_to_host ( V_T );

                for ( long i=0; i<length; i++ )
                    { v.ptr[i] = static_cast <host_precision> ( V_T.ptr[i] ); }
            }
        }


        template<typename host_precision>
        void copy_to_host ( run_time_vector<host_precision> & v ) const
        {
            if constexpr ( std::is_same_v< host_precision , T > )
                { mem_copy_to_host ( v ); }
            else 
                { val_copy_to_host <host_precision> (v); }
        }



        void make_shallow_copy ( cuda_run_time_vector_shallow_copy <T> & v ) const
        {
            v.length = this->length;
            v.ptr    = this->ptr;
        }



// // The following syntax needs testing; what's the difference in effect between && and &, and no & at all?
//         cuda_run_time_vector_shallow_copy <T> && make_shallow_copy ()
//         {
//             cuda_run_time_vector_shallow_copy <T> v;

//             v.length = this->length;
//             v.ptr    = this->ptr;

//             return v;
//         }
        // [2023/07/11]
        // NOTE: Did we ever use the above function? If not, we may consider comment it out.



        // copy ctor
        cuda_run_time_vector( cuda_run_time_vector<T> const & v ) : cuda_run_time_vector ( v.length ) // : delegating to user-supplied ctor
            { copy (v); }

        // copy assignment operator
        cuda_run_time_vector<T> & operator=( cuda_run_time_vector<T> const & v )
            { copy (v); return *this; }

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


        void memset_zero() 
            { cudaMemset ( ptr, 0, length * sizeof(T) ); }
}; 
//class cuda_run_time_vector



template<typename T>
class cuda_run_time_matrix
{
    public:
        long rows = -1<<30;
        long cols = -1<<30;
        T * ptr = nullptr;


        std::string name = "undefined";


        cuda_run_time_matrix(){} 


        void allocate_memory(long const M , long const N)
        {
            if ( ptr != nullptr ) 
                { printf("Error: ptr is already allocated - cuda_run_time_matrix.\n"); fflush(stdout); exit(0); }

            if (M <= 0 || N <= 0) 
                { printf("Error: rows/cols <= 0 - allocate_memory - cuda_run_time_matrix.\n"); fflush(stdout); exit(0); }
            this->rows = M;
            this->cols = N;

// need to check how big is the memory being asked as in cuda_run_time_vector

            cudaError_t malloc_result = cudaMalloc( &ptr, M*N*sizeof(T) );
            if ( malloc_result != cudaSuccess ) 
                { printf( "cudaMalloc unsuccessful: %s.\n", cudaGetErrorString(malloc_result) ); exit(0); }
            cudaMemset ( ptr, 0, M*N*sizeof(T) );        // Memory is not cleared with cudaMalloc;
            // NOTE: cudaMalloc should account for alignment.
        }
        // NOTE: If empty constructor (currently the default constructor) is called, the above
        //       function needs to be called explicitly to allocate memory.


        cuda_run_time_matrix(long const M , long const N) // a user-supplied 'non-speical' constructor
            { allocate_memory(M , N); }


        void copy ( cuda_run_time_matrix<T> const & A )
        {
            if ( this == &A || this->ptr == A.ptr ) 
                { printf("Error: self copy - copy - cuda_run_time_matrix.\n"); fflush(stdout); exit(0); }
            
            if ( this->ptr == nullptr || A.ptr == nullptr ) 
                { printf("Error: nullptr   - copy - cuda_run_time_matrix.\n"); fflush(stdout); exit(0); }

            if ( rows <= 0 || cols <= 0 || rows != A.rows || cols != A.cols ) 
                { printf("Error: length    - copy - cuda_run_time_matrix.\n"); fflush(stdout); exit(0); }

            cudaError_t memcpy_result = cudaMemcpy ( ptr , A.ptr , A.rows * A.cols * sizeof(T) , cudaMemcpyDeviceToDevice );
            if ( memcpy_result != cudaSuccess ) 
                { printf( "cudaMemcpy unsuccessful %s.\n", cudaGetErrorString(memcpy_result) ); exit(0); }
        }
        // [2023/07/11]
        // NOTE: In "copy ()", we copy from "device" to "device", which is the difference from 
        //       "_copy_from_host ()" and "_copy_to_host ()".

        void mem_copy_from_host ( run_time_matrix<T> const & A )
        {
            if ( this->ptr == nullptr || A.ptr == nullptr ) 
                { printf("%s %d Error:   nullptr.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            if ( rows <= 0 || cols <= 0 || rows != A.rows || cols != A.cols ) 
                { printf("%s %d Error:    length.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            cudaError_t memcpy_result = cudaMemcpy ( ptr , A.ptr , A.rows * A.cols * sizeof(T) , cudaMemcpyHostToDevice );
            if ( memcpy_result != cudaSuccess ) 
            { 
                printf( "(%s) %s %d cudaMemcpy unsuccessful for %s.\n", cudaGetErrorString(memcpy_result),
                        __FILE__, __LINE__, this->name.c_str() ); fflush(stdout); exit(0); 
            }
        }


        template<typename host_precision>
        void val_copy_from_host ( run_time_matrix<host_precision> const & A )
        {
            if ( this->ptr == nullptr || A.ptr == nullptr ) 
                { printf("%s %d Error:   nullptr.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            if ( rows <= 0 || cols <= 0 || rows != A.rows || cols != A.cols ) 
                { printf("%s %d Error:    length.\n", __FILE__, __LINE__); fflush(stdout); exit(0); }

            if constexpr ( std::is_same_v< host_precision , T > )
            {
                printf("\n\n%s %d consider using mem_copy_from_host directly.\n\n", __FILE__, __LINE__);
                mem_copy_from_host ( A );
            }
            else 
            {
                run_time_matrix<T> A_T (A.rows,A.cols);
                for ( long i=0; i<rows; i++ ) 
                    for ( long j=0; j<cols; j++ ) 
                        { A_T.ptr[ i * cols + j ] = static_cast <T> ( A.ptr[ i * cols + j ] ); }

                mem_copy_from_host (A_T);
            }
        }


        template<typename host_precision>
        void copy_from_host ( run_time_matrix<host_precision> const & A )
        {
            if constexpr ( std::is_same_v< host_precision , T > )
                { mem_copy_from_host ( A ); }
            else 
                { val_copy_from_host <host_precision> ( A ); }
        }


        void make_shallow_copy ( cuda_run_time_matrix_shallow_copy <T> & A ) const
        {
            A.rows = this->rows;
            A.cols = this->cols;
            A.ptr  = this->ptr;
        }
        
        // copy ctor
        cuda_run_time_matrix( cuda_run_time_matrix<T> const & A ) : cuda_run_time_matrix ( A.rows , A.cols ) // : delegating to user-supplied ctor
            { copy (A); }
        
        // copy assignment operator
        cuda_run_time_matrix<T> & operator=( cuda_run_time_matrix<T> const & A )
            { copy (A); return *this; }

        // move ctor
        cuda_run_time_matrix( cuda_run_time_matrix<T> && A ) noexcept
        {
            this->rows = A.rows;  this->cols = A.cols;  this->ptr = A.ptr;

            A.rows = 0;           A.cols = 0;           A.ptr = nullptr;
        }

        // move assignment operator
        cuda_run_time_matrix<T> & operator=( cuda_run_time_matrix<T> && v ) = delete;
        // NOE: For rationale behind '= delete', see the comments in folder rule_035.


        // dtor
        ~cuda_run_time_matrix() 
        { 
            cudaError_t free_result = cudaFree(ptr); 
            if ( free_result != cudaSuccess ) 
                { printf("cudaFree unsuccessful: %s.\n", cudaGetErrorString(free_result)); exit(0); }
        }
        // Questions: Is cudaFree(ptr) enough? What might be the corner cases that I haven't thought of?


        void memset_zero() 
            { cudaMemset ( ptr, 0, rows * cols * sizeof(T) ); }
};
// class cuda_run_time_matrix


template<typename T>
class cuda_run_time_vector_shallow_copy
{
    public:
        long length = -1<<30;
        T * ptr = nullptr;

        __device__ T& operator() (long const i)       { return ptr[i]; } // read and write access
        __device__ T& operator() (long const i) const { return ptr[i]; } // read           access
}; 
//class cuda_run_time_vector_shallow_copy


template<typename T>
class cuda_run_time_matrix_shallow_copy
{
    public:
        long rows = -1<<30;
        long cols = -1<<30;
        T * ptr = nullptr;

        __device__ T& operator() (long const i , long const j)       { return ptr[ i * cols + j ]; } // read and write access
        __device__ T& operator() (long const i , long const j) const { return ptr[ i * cols + j ]; } // read           access
};
// class cuda_run_time_matrix_shallow_copy


#endif