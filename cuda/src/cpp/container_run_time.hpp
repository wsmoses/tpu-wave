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
            if ( ptr != nullptr ) { printf("Error: ptr is already allocated - run_time_vector.\n"); fflush(stdout); exit(0); }

            if (L <= 0) { printf("Error: length <= 0 - allocate_memory - run_time_vector.\n"); fflush(stdout); exit(0); }
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
            if ( ptr == nullptr ) { printf("Error: aligned_alloc returned a nullptr.\n"); fflush(stdout); exit(0); }
            memset ( ptr, 0, byte_size );
        }
        // NOTE: If empty constructor (currently the default constructor) is called, the above
        //       function needs to be called explicitly to allocate memory.
        //
        // NOTE: Do we want to make 'alignment' in the above member function a global variable 
        //       (possibly defined in some other file) or not? 
        //           I think leave it here is better - locality, hence more expressive.


        run_time_vector (int const L) // a user-supplied 'non-speical' constructor
            { allocate_memory (L); }
            

        void copy ( run_time_vector<T> const & v )
        {
            // NOTE: Syntactically, this copy member function should 'work' for self assignment. Its
            //       behavior is (arguably) 'logically' correct. However, such 'self' assignment is 
            //       an indication that some error or misuse may have occurred from the 'application'
            //       side. We add this test (test_self_copy) to guard such misuse.
            if ( this == &v || this->ptr == v.ptr ) 
                { printf("Error: self copy - copy - run_time_vector.\n"); fflush(stdout); exit(0); }
            
            if ( this->ptr == nullptr || v.ptr == nullptr ) 
                { printf("Error: nullptr   - copy - run_time_vector.\n"); fflush(stdout); exit(0); }

            if ( length <= 0 || length != v.length ) 
                { printf("Error: length    - copy - run_time_vector.\n"); fflush(stdout); exit(0); }

            for ( int i=0; i<length; i++ ) { ptr[i] = v.ptr[i]; }   // { this->at(i) = v.at(i); }
        }


        // copy ctor
        run_time_vector( run_time_vector<T> const & v ) : run_time_vector ( v.length ) // : delegating to user-supplied ctor
            { copy (v); }

        // copy assignment operator
        run_time_vector<T> & operator=( run_time_vector<T> const & v )
            { copy (v); return *this; }

        // move ctor
        run_time_vector( run_time_vector<T> && v ) noexcept
        {
            this->length = v.length;  this->ptr = v.ptr;

            v.length = 0;             v.ptr = nullptr;
        }
        // NOTE: I think the above implementation is 'safe'; move ctor does not call allocate_memory (),
        //       therefore we can directly assign to this->ptr and set v.ptr to nullptr. 

        // move assignment operator
        run_time_vector<T> & operator=( run_time_vector<T> && v ) = delete;
        // NOTE: For rationale behind '= delete', see the comments in folder rule_035.


        // dtor
        ~run_time_vector() { free(ptr); }
        // Questions: Is free(ptr) enough? What might be the corner cases that I haven't thought of?


        // assignment operators
        void operator= ( std::initializer_list<T> il ) 
        { 
            if ( static_cast<long unsigned int>(length) != il.size() ) 
                { printf("Error: length - operator= - run_time_vector.\n"); fflush(stdout); exit(0); }

            int i = 0; for ( auto& e : il ) { this->at(i) = e; i++; }
        }

        void operator*= (T const &a) { for ( int i=0; i<length; i++ ) { ptr[i] = (double) ptr[i] * (double) a; } } 
                                                      // { this->operator()(i) *= a; }

        void operator/= (T const &a) { for ( int i=0; i<length; i++ ) { ptr[i] = (double) ptr[i] / (double) a; } }
                                                      // { this->operator()(i) /= a; }

        // member access
        T* data() { return ptr; }

        T& operator() (int const i) // read and write access
        { 
            if ( ptr == nullptr   ) { printf("Error: nullptr - operator() - run_time_vector.\n"); fflush(stdout); exit(0); }
            if ( i == 359999 ) { printf("DEBUG op: i=%d, length=%d\n", i, length); fflush(stdout); }
            if ( i<0 || i>=length ) { printf("Out of bound access - operator() - run_time_vector. i=%d, length=%d\n", i, length);   
                                      printf("    Possible cue 1): Forgot to change the input parameters in compile_time_namespace.hpp?\n");
                                      printf("    Possible cue 2): Forgot to change the medium name/path in run time command?\n"); fflush(stdout); exit(0); }

            return ptr[i]; 
        }

        T& at (int const i) // read and write access
        {
            if ( ptr == nullptr   ) { printf("Error: nullptr - at - run_time_vector.\n"); fflush(stdout); exit(0); }
            if ( i == 359999 ) { printf("DEBUG at: i=%d, length=%d\n", i, length); fflush(stdout); }
            if ( i<0 || i>=length ) { printf("Out of bound access - at - run_time_vector. i=%d, length=%d\n", i, length);   
                                      printf("    Possible cue 1): Forgot to change the input parameters in compile_time_namespace.hpp?\n");
                                      printf("    Possible cue 2): Forgot to change the medium name/path in run time command?\n"); fflush(stdout); exit(0); }

            return ptr[i];
        }


        void set_constant(T const & a) { for ( int i = 0; i < length; i++ ) { ptr[i] = a; } }


        template<int L>
        void copy ( compile_time_vector<T,L> const &input_vector ) // copy from a compile_time_vector
        {
            if ( this->length != L ) 
                { printf("Error: length - copy (compile_time_vector) - run_time_vector.\n"); fflush(stdout); exit(0); }

            for ( int i=0; i<length; i++ ) { this->at(i) = input_vector.at(i); }
        }
        // NOTE: the above member function copies from a compiler_time_vector, whose member function 'at()' is constexpr.


        // // NOTE: In the following member function, (effectively,) the result of a * x + y is ASSIGNED
        // //       to this (calling) run_time_vector.
        // //           Both x and y are passed in as CONST references. When using this member function,
        // //       the (calling) run_time_vector may be passed in as x or y (particularly y). In this 
        // //       case, some thoughts are required to understand the CONST qualifier - it means the 
        // //       referenced object cannot be modified THROUGH this (const) reference; it can still be 
        // //       modified through other acesses, such as through the calling vector itself. 
        // //           Syntactically, passing the calling run_time_vector itself as either x or y in the
        // //       following member function should be CORRECT.
        // void axpy_assign ( T const &a , run_time_vector<T> const &x , run_time_vector<T> const &y ) 
        // { 
        //     if ( this->length != x.length || this->length != y.length ) 
        //         { printf("Error: length - axpy_assign - run_time_vector.\n"); fflush(stdout); exit(0); }

        //     for ( int i=0; i<this->length; i++ ) { this->ptr[i] = a * x.ptr[i] + y.ptr[i]; }
        //                                       // { this->at (i) = a * x    (i) + y    (i); }
        // }
        // // axpy_assign () /* not used currently */
        // // axpy_add_to () /* not defined yet; change = to += */


        // NOTE: As a special case of the above three-parameter member function axpy_assign (), a 
        //       two-parameter version is defined below for the situation where y is the calling 
        //       run_time_vector itself.
        //           Not sure whether this version will simplify the 'job' for the complier or not
        //       (save a copy operation or some temporary storage, for example); but, at least, by 
        //       removing a redundant input argument, we lower the risk of inadvertent misuse.
        void ax_add_to ( T const &a , run_time_vector<T> const &x ) 
        { 
            if ( this->length != x.length ) 
                { printf("Error: length - ax_add_to - run_time_vector.\n"); fflush(stdout); exit(0); }

            for ( int i=0; i<this->length; i++ ) { this->ptr[i] += a * x.ptr[i]; }
                                              // { this->at (i) += a * x    (i); }           
        }
        // ax_add_to ()
        // ax_assign /* not defined yet; change += to = */
        


        void ax_by_assign ( T const &a , run_time_vector<T> const &x ,
                            T const &b , run_time_vector<T> const &y ) 
        { 
            if ( this->length != x.length || this->length != y.length ) 
                { printf("Error: length - ax_by_assign - run_time_vector.\n"); fflush(stdout); exit(0); }

            for ( int i=0; i<this->length; i++ ) { this->ptr[i] = a * x.ptr[i] + b * y.ptr[i]; }
                                              // { this->at (i) = a * x    (i) + b * y    (i); }
        }
        // ax_by_assign ()
        // a very "general" function, may be less efficient than the possibly 
        // more specialized versions (such as scale_self_then_add_b)



        void retrieve_row ( int const i_row , run_time_matrix<T> const & M ) 
        {
            if ( this->length != M.cols ) 
                { printf("Error: lengths mismatch - retrieve_row - run_time_vector.\n"); fflush(stdout); exit(0); }

            if ( i_row < 0 || i_row >= M.rows ) 
                { printf("Error: row out of range - retrieve_row - run_time_vector.\n"); fflush(stdout); exit(0); }

            for ( int j=0; j<length; j++ ) { this->ptr[j] = M.ptr[ i_row * M.cols + j ]; } 
                                        // { this->operator()(j) = M(i_row,j); } 
        }
        // NOTE: Argument 'i_row' is passed in as 'int const' so that it won't be changed inadvertently.
        //       /* Risk of such 'inadvertent change' would be higher if the input argument name is i,
        //          since j is loop index and changes within the loop. */


        T L2_norm () const
            { return sqrt( L2_norm_square () ); }  // Question: will sqrt from math.h work well with template ?

        T L2_norm_square () const
        {
            T norm_square = static_cast<T>(0);
            for ( int i=0; i<length; i++ ) 
                { norm_square += this->ptr[i] * this->ptr[i]; }

            return norm_square;
        }

        void print ()
        {   
            // printf( "length : %d ; ptr : %p \n" , this->length , (void *)(this->ptr) );
            for ( int i=0; i<length; i++ ) 
                // { std::cout << this->at(i) << " "; } printf( "\n" );
                { printf("% 16.15e \n" , static_cast< double > ( this->at(i) ) ); }
            printf( "\n" );
        }

        void print ( int const bgn , int const end ) const    // NOTE: "end" is NOT inclusive
        {   
            // printf( "length : %d \n" , end - bgn );
            for ( int i=bgn; i<end; i++ ) 
                { printf("% 16.15e \n" , static_cast< double > ( this->ptr[i] ) ); }
        }

        void print_for_copy ( int const bgn , int const end ) const    // NOTE: "end" is NOT inclusive
        {   
            printf( "length : %d \n" , end - bgn );
            printf( "{ { " );
            for ( int i=bgn; i<end; i++ ) 
            { 
                printf("% 16.15e" , static_cast< double > ( this->ptr[i] ) ); 
                if ( i < end-1 ) { printf(" , "); }
            }
            printf( " } };\n\n" );
        }

        // NOTE: Add begin () and end () so that we can use the utility functions provided 
        //       in <algorithm>.
        //           Be careful with the return type of functions such as std::max_element,
        //       which is an 'iterator', or rather a pointer in this particular case. Need
        //       to dereference it to retrieve the value.
        //
        //       The following using or typedef statements seem to be not needed, but may 
        //       provide better type safety.
        //           using       iterator =       T *;    // typedef T * iterator;
        //           using const_iterator = const T *;    // typedef const T* const_iterator;
        // 
        T* begin () { return ptr;          }
        T* end   () { return ptr + length; } // 1 past the last
}; 
//class run_time_vector


template<typename T>
static inline T run_time_vector_inner_product ( run_time_vector<T> const & v1 , run_time_vector<T> const & v2 )
{
    if ( v1.length <= 0 || v2.length <= 0 || v1.length != v2.length ) 
        { printf("Error: length - inner_product - run_time_vector.\n"); fflush(stdout); exit(0); }
    
    T inner_product = static_cast<T>(0);
    for ( int i = 0; i < v1.length; i++ ) { inner_product += v1.ptr[i] * v2.ptr[i]; }
    
    return inner_product;
}



template<typename T>
class run_time_matrix
{
    public:
        int rows = -1<<30;
        int cols = -1<<30;
        T * ptr = nullptr;

        run_time_matrix(){} 


        void allocate_memory(int const M , int const N)
        {
            if ( ptr != nullptr ) { printf("Error: ptr is already allocated - run_time_matrix.\n"); fflush(stdout); exit(0); }

            if (M <= 0 || N <= 0) { printf("Error: rows/cols <= 0 - allocate_memory - run_time_matrix.\n"); fflush(stdout); exit(0); }
            this->rows = M;
            this->cols = N;

            constexpr int alignment = 32;    // 32 bytes; 
            int byte_size = ( ( M*N*sizeof(T) + alignment - 1 ) / alignment ) * alignment;

            ptr = static_cast<T *>( aligned_alloc(alignment, byte_size) );
            if ( ptr == nullptr ) { printf("Error: aligned_alloc returned a nullptr.\n"); fflush(stdout); exit(0); }
            memset ( ptr, 0, byte_size );
        }
        // NOTE: If empty constructor (currently the default constructor) is called, the above
        //       function needs to be called explicitly to allocate memory.


        run_time_matrix(int const M , int const N) // a user-supplied 'non-speical' constructor
            { allocate_memory(M , N); }


        void copy ( run_time_matrix<T> const & A )
        {
            if ( this == &A || this->ptr == A.ptr ) 
                { printf("Error: self copy - copy - run_time_matrix.\n"); fflush(stdout); exit(0); }
            
            if ( this->ptr == nullptr || A.ptr == nullptr ) 
                { printf("Error: nullptr   - copy - run_time_matrix.\n"); fflush(stdout); exit(0); }

            if ( rows <= 0 || cols <= 0 || rows != A.rows || cols != A.cols ) 
                { printf("Error: length    - copy - run_time_matrix.\n"); fflush(stdout); exit(0); }

            for ( int i=0; i<rows; i++ ) 
                for ( int j=0; j<cols; j++ ) 
                    { ptr[ i * cols + j ] = A.ptr[ i * cols + j ]; }
        }

        
        // copy ctor
        run_time_matrix( run_time_matrix<T> const & A ) : run_time_matrix ( A.rows , A.cols ) // : delegating to user-supplied ctor
            { copy (A); }
        
        // copy assignment operator
        run_time_matrix<T> & operator=( run_time_matrix<T> const & A )
            { copy (A); return *this; }

        // move ctor
        run_time_matrix( run_time_matrix<T> && A ) noexcept
        {
            this->rows = A.rows;  this->cols = A.cols;  this->ptr = A.ptr;

            A.rows = 0;           A.cols = 0;           A.ptr = nullptr;
        }

        // move assignment operator
        run_time_matrix<T> & operator=( run_time_matrix<T> && v ) = delete;
        // NOE: For rationale behind '= delete', see the comments in folder rule_035.


        // dtor
        ~run_time_matrix() { free(ptr); }
        // Questions: Is free(ptr) enough? What might be the corner cases that I haven't thought of?


        // assignment operators
        void operator*= (T const &a)
        { 
            for ( int i=0; i<rows; i++ ) 
                for ( int j=0; j<cols; j++ ) { ptr[ i * cols + j ] = (double) ptr[ i * cols + j ] * (double) a; } 
                                        // { this->operator()(i,j) *= a; }
        }

        void operator/= (T const &a)
        { 
            for ( int i=0; i<rows; i++ ) 
                for ( int j=0; j<cols; j++ ) { ptr[ i * cols + j ] = (double) ptr[ i * cols + j ] / (double) a; }
                                        // { this->operator()(i,j) /= a; }
        }

        // member access
        T* data() { return ptr; }

        T& operator() (int const i , int const j) // read and write access
        { 
            // if ( ptr == nullptr ) { printf("Error: nullptr - operator() - run_time_matrix.  \n"); fflush(stdout); exit(0); }
            // if ( i<0 || i>=rows ) { printf("Row out of bound - operator() - run_time_matrix.\n"); fflush(stdout); exit(0); }
            // if ( j<0 || j>=cols ) { printf("Col out of bound - operator() - run_time_matrix.\n"); fflush(stdout); exit(0); }

            return ptr[ i * cols + j ]; 
        }
        // NOTE: We can overload the above function as T& operator() (int i , int j) const { return ptr[i*cols+j]; }
        //       so that member functions such as axpy that takes an input argument run_time_matrix<T> const &x can 
        //       access its 'data element' via x(i,j), which is less error-prone and easier to maintain compared to
        //       using the 'subscript' operator []. However, the current view is to not 'make the switch' due to 
        //       concern on the compiler's ability to 'optimize' (in particular, 'vectorize') through the member 
        //       function. More detailed comments can be found in folder const_overload.

        T& at (int const i , int const j) // read and write access
        {
            if ( ptr == nullptr ) { printf("Error: nullptr - at - run_time_matrix.  \n"); fflush(stdout); exit(0); }
            if ( i<0 || i>=rows ) { printf("Row out of bound - at - run_time_matrix.\n"); fflush(stdout); exit(0); }
            if ( j<0 || j>=cols ) { printf("Col out of bound - at - run_time_matrix.\n"); fflush(stdout); exit(0); }

            return ptr[ i * cols + j ];
        }


        void set_constant(T const &a)
        { 
            for ( int i=0; i<rows; i++ ) 
                for ( int j=0; j<cols; j++ ) { ptr[ i * cols + j ] = a; } 
                                        // { this->operator()(i,j) = a; } 
        }


        template<int M, int N>
        void copy ( compile_time_matrix<T,M,N> const &A )
        {
            if ( rows != M || cols != N ) 
                { printf("Error: length - copy (compile_time_matrix) - run_time_matrix.\n"); fflush(stdout); exit(0); }

            for ( int i=0; i<rows; i++ ) 
                for ( int j=0; j<cols; j++ ) { this->at(i,j) = A.at(i,j); }
        }


        // // NOTE:  /* See the same comments in run_time_vector. */
        // void axpy_assign ( T const &a , run_time_matrix<T> const &x , run_time_matrix<T> const &y ) 
        // { 
        //     if ( rows != x.rows || rows != y.rows ) 
        //         { printf("Error: rows - axpy_assign - run_time_matrix.\n"); fflush(stdout); exit(0); }
        //     if ( cols != x.cols || cols != y.cols ) 
        //         { printf("Error: cols - axpy_assign - run_time_matrix.\n"); fflush(stdout); exit(0); }

        //     for ( int i=0; i<rows; i++ ) 
        //         for ( int j=0; j<cols; j++ ) 
        //             { ptr[ i * cols + j ] = a * x.ptr[ i * cols + j ] + y.ptr[ i * cols + j ]; }
        // }
        // // axpy_assign () /* not used currently */
        // // axpy_add_to () /* not defined yet; change = to += */


        // NOTE:  /* See the same comments in run_time_vector. */
        void ax_add_to ( T const &a , run_time_matrix<T> const &x ) 
        { 
            if ( rows != x.rows ) 
                { printf("Error: rows - ax_add_to - run_time_matrix.\n"); fflush(stdout); exit(0); }
            if ( cols != x.cols ) 
                { printf("Error: cols - ax_add_to - run_time_matrix.\n"); fflush(stdout); exit(0); }

            for ( int i=0; i<rows; i++ ) 
                for ( int j=0; j<cols; j++ ) 
                    { ptr[ i * cols + j ] += a * x.ptr[ i * cols + j ]; }
        }
        // ax_add_to ()
        // ax_assign /* not defined yet; change += to = */


        void subtract_then_scale ( run_time_matrix<T> const &x , T const &a ) 
        { 
            if ( rows != x.rows ) 
                { printf("Error: rows - ax_add_to - run_time_matrix.\n"); fflush(stdout); exit(0); }
            if ( cols != x.cols ) 
                { printf("Error: cols - ax_add_to - run_time_matrix.\n"); fflush(stdout); exit(0); }

            for ( int i=0; i<rows; i++ ) {
                for ( int j=0; j<cols; j++ ) { 
                    ptr[ i * cols + j ] -= x.ptr[ i * cols + j ]; 
                    ptr[ i * cols + j ] *= a;
                }
            }
        }
        // ax_add_to ()
        // ax_assign /* not defined yet; change += to = */



        void fill_row ( int const i_row , run_time_vector<T> & v )
        {
            if ( cols != v.length ) 
                { printf("Error: length - fill_row - run_time_matrix.\n"); fflush(stdout); exit(0); }

            for ( int j=0; j<cols; j++ ) { ptr[ i_row * cols + j ] = v.ptr[j]; }
                                    // { this->operator()(i_row,j) = v    (j); } 
        }
        
        void print ()
        {
            printf( "rows : %d ; cols : %d ; ptr : %p \n" , rows , cols , (void *)(this->ptr) );
            for ( int i=0; i<rows; i++ ) {
                for ( int j=0; j<cols; j++ ) { 
                    // std::cout << this->at(i,j) << " "; 
                    printf("% 16.15e " , static_cast< double > ( this->at(i,j) ) );
                } std::cout << "\n";
            } std::cout << "\n";
        }

        void print_for_copy ()
        {
            printf( "rows : %d ; cols : %d ; ptr : %p \n" , rows , cols , (void *)(this->ptr) );
            printf( "{ { \n" );
            for ( int i=0; i<rows; i++ ) 
            {
                printf( "    { " );
                for ( int j=0; j<cols; j++ ) 
                { 
                    printf("% 16.15e" , static_cast< double > ( this->at(i,j) ) ); 
                    if ( j < cols-1 ) { printf(" , "); }
                } 
                printf( " }" );
                if ( i < rows-1 ) { printf(" , \n"); }
            } 
            printf( "\n} };\n\n" );
        }



        // NOTE: use of std::algorithm based on the following begin () and end () may lead to 
        //       'logically' unexpected results; haven't thought them through yet; 
        // T* begin () { return ptr;               }
        // T* end   () { return ptr + rows * cols; } // 1 past the last
};
// class run_time_matrix


template<typename T>
static inline T run_time_matrix_inner_product ( run_time_matrix<T> const & A1 , run_time_matrix<T> const & A2 )
{
    if ( A1.rows <= 0 || A2.rows <= 0 || A1.rows != A2.rows ) 
        { printf("Error: rows - inner_product - run_time_matrix.\n"); fflush(stdout); exit(0); }
    if ( A1.cols <= 0 || A2.cols <= 0 || A1.cols != A2.cols ) 
        { printf("Error: cols - inner_product - run_time_matrix.\n"); fflush(stdout); exit(0); }
    
    T inner_product = static_cast<T>(0);
    for ( int i=0; i<A1.rows; i++ ) 
        for ( int j=0; j<A1.cols; j++ ) 
            { inner_product += A1.ptr[ i * A1.cols + j ] * A2.ptr[ i * A2.cols + j ]; }
    
    return inner_product;
}



// NOTE: I thought std::aligned_alloc is not recognized by clang, but it turns out it is 
//       not recognized by 'APPLE(!!!) clang' - could have done the memory allocation in
//       allocate_memory () differently, if not for the fear of compatibility issues when
//       switching between different compilers.
//           Well, in this particular case, std::aligned_alloc and std::free are probably
//       (with a good confidence) just wrappers around their C equivalents (same name but 
//       without the std scope qualifier; defined in stdlib.h rather than cstdlib), so not
//       much is missed out. 
// 
//           But let this be a WARNING and learn from it! 
// 
//           I could have missed some functionality that may have simplified things for me, 
//       which is already incorporated in the c++ language design and supported by g++ and 
//       clang, but not yet supported or disabled by 'APPLE(!!!) clang'.


#endif