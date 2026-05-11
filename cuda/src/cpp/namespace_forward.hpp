#ifndef NAMESPACE_FORWARD_H
#define NAMESPACE_FORWARD_H

#include <array>
#include <map>

#include "container_compile_time.hpp"
#include "namespace_type.cuh"

namespace namespace_forward
{
    constexpr int N_dir = 2;

    constexpr int length_stencil = 4;

    // WARNING: The following enum definitions will "pollute" the namespace, e.g., L can be used directly, 
    //          which will be interpreted as referring to enum_side::L defined in the namespace.
    enum enum_dir  { x , y };
    enum enum_type { M , N };
    enum enum_side { L , R };

    constexpr enum_type operator~(const enum_type g_type)
    {
        if ( g_type == enum_type::N ) return enum_type::M;
        if ( g_type == enum_type::M ) return enum_type::N;
        else { printf("g_type is not an enum_type.\n"); exit(0); }
    }

    constexpr char enum_to_char(const enum_type g_type)
    {
        if ( g_type == enum_type::N ) return 'N';
        if ( g_type == enum_type::M ) return 'M';
        else { printf("g_type is not an enum_type.\n"); exit(0); }
    }

    constexpr enum_type char_to_enum(const char g_type)
    {
        if ( g_type == 'N' ) return enum_type::N;
        if ( g_type == 'M' ) return enum_type::M;
        else { printf("g_type can only be N or M.\n"); exit(0); }
    }


// NOTE: The following function maps the physical direction ('X' or 'Y') to data direction (0 or 1).
//       The physical direction is used to interface with "application" side, i.e., specification of
//       the sizes, src and rcv locations, etc.
//           The data direction is used to interface with "implementation" side, i.e., how the 1D 
//       data should be interpreted. On the "implementation" side, such as grid, the lower letter 
//       'x', 'y', and 'z' are used synonymously with 0, 1, and 2, meaning the slowest to fastest
//       data direction.
//           This distinction between {'X','Y','Z'} and {'x','y','z'} can be confusing.
//          
//           The following function needs to be called in "input_processing.cpp" and "inverse.cpp" 
//       (where the src and rcv are processed).

constexpr int XY_to_01(const char c_dir)
{
         if ( c_dir == 'X' ) return 1;
    else if ( c_dir == 'Y' ) return 0;
    else { printf("c_dir can only be X or Y.\n"); exit(0); }
}
// NOTE: This mapping may change if decided to change direction.


    constexpr compile_time_vector<double,4> stencil { { 1./24 , -9./8 , 9./8 , -1./24 } };
    // inline compile_time_vector<double,4> stencil { { 1./24 , -9./8 , 9./8 , -1./24 } };
    // [2023/05/13]
    // NOTE: Some conversion function is implicitly called for __half, which is not constexpr.
    // [2024/04/03]
    // NOTE: constexpr would work here because we are no longer using __half here.

    static_assert ( length_stencil == stencil.length , "stencil length" );


    // // "Operators" used in secure_bdry
    // struct struct_S_extr 
    // { 
    //     static inline compile_time_matrix<ns_type::host_precision,2,2> DM_L { { { stencil(0), stencil(1) } ,
    //                                                                         { 0.,         stencil(0) } } };
    //     static inline compile_time_matrix<ns_type::host_precision,2,2> DM_R { { { stencil(3), 0.         } ,
    //                                                                         { stencil(2), stencil(3) } } };

    //     static inline compile_time_matrix<ns_type::host_precision,1,1> DN_L { { { stencil(0) } } };
    //     static inline compile_time_matrix<ns_type::host_precision,1,1> DN_R { { { stencil(3) } } };
    // } 
    // constexpr S_extr;
    // 
    //       
    // // "Operators" used in apply_stencil_*
    // struct struct_S_intr 
    // { 
    //     static inline compile_time_matrix<ns_type::host_precision,2,3> DM_L { { { stencil(2), stencil(3), 0.         } ,
    //                                                                           { stencil(1), stencil(2), stencil(3) } } };
    //     static inline compile_time_matrix<ns_type::host_precision,2,3> DM_R { { { stencil(0), stencil(1), stencil(2) } ,
    //                                                                           { 0.,         stencil(0), stencil(1) } } };

    //     static inline compile_time_matrix<ns_type::host_precision,1,3> DN_L { { { stencil(1), stencil(2), stencil(3) } } };
    //     static inline compile_time_matrix<ns_type::host_precision,1,3> DN_R { { { stencil(0), stencil(1), stencil(2) } } };
    // } 
    // constexpr S_intr; 
    //       
    //       
    // [2024/04/02]
    // NOTE: "S_extr" and "S_intr" are only needed when having MPI interfaces (or when treating 
    //       the periodic boundary as an MPI interface). They are not really used currently and 
    //       are commented out to help focus.
        

    // "Operators" used in apply_stencil_*
    struct struct_W_intr 
    { 
        static inline compile_time_matrix<double,4,5> DM_L { { { -2.       ,  3.       , -1.       ,  0.       ,  0.       } ,
                                                               { -1.       ,  1.       ,  0.       ,  0.       ,  0.       } ,
                                                               {  1. / 24. , -9. / 8.  ,  9. / 8.  , -1. / 24. ,  0.       } ,
                                                               { -1. / 71. ,  6. / 71. , -83./ 71. ,  81./ 71. , -3. / 71. } } };

        static inline compile_time_matrix<double,4,5> DM_R { { {  3. / 71. , -81./ 71. ,  83./ 71. , -6. / 71. ,  1. / 71. } ,
                                                               {  0.       ,  1. / 24. , -9. / 8.  ,  9. / 8.  , -1. / 24. } ,
                                                               {  0.       ,  0.       ,  0.       , -1.       ,  1.       } ,
                                                               {  0.       ,  0.       ,  1.       , -3.       ,  2.       } } };

        static inline compile_time_matrix<double,3,5> DN_L { { {-79. / 78. ,  27./ 26. , -1. / 26. ,  1. / 78. ,  0.       } ,
                                                               { 2.  / 21. , -9. / 7.  ,  9. / 7.  , -2. / 21. ,  0.       } ,
                                                               { 1.  / 75. ,  0.       , -27./ 25. ,  83./ 75. , -1. / 25. } } };

        static inline compile_time_matrix<double,3,5> DN_R { { {  1. / 25. , -83./ 75. ,  27./ 25. ,  0.       , -1. / 75. } ,
                                                               {  0.       ,  2. / 21. , -9. / 7.  ,  9. / 7.  , -2. / 21. } ,
                                                               {  0.       , -1. / 78. ,  1. / 26. , -27./ 26. ,  79./ 78. } } };
    } 
    constexpr W_intr;  // weak bdry only involves interior "stencils"


    // cannot constexpr map? an alternative is to use std::array with N and M decay to 0 and 1
    // std::map<char,int> W_intr_rows{ std::pair('N',4) , std::pair('M',3) };
    // std::map<char,int> W_intr_cols{ std::pair('N',5) , std::pair('M',5) };

    struct struct_A_diag 
    {
        static inline compile_time_vector< double , 4 > N_L { { 7./18.  , 9./8. , 1.    , 71./72. } };
        static inline compile_time_vector< double , 4 > N_R { { 71./72. , 1.    , 9./8. , 7./18.  } };
        
        static inline compile_time_vector< double , 3 > M_L { { 13./12. ,  7./8.  ,  25./24. } };
        static inline compile_time_vector< double , 3 > M_R { { 25./24. ,  7./8.  ,  13./12. } };
    } 
    constexpr A_diag; // A is only used in the weak context


    struct struct_projection_operator
    {
        static inline compile_time_vector< double , 1 > N_L { { 1. } };
        static inline compile_time_vector< double , 1 > N_R { { 1. } };
        
        static inline compile_time_vector< double , 3 > M_L { { 15./8. , -5./4. ,  3./8. } };
        static inline compile_time_vector< double , 3 > M_R { {  3./8. , -5./4. , 15./8. } };
    } 
    constexpr projection_operator;

    // constexpr auto define_Array_Grid_types ()
    consteval auto define_Array_Grid_types ()    // if some compiler versions complain, 
    {                                            // try changing consteval to constexpr
        std::array< std::array<char, N_dir> , 2<<(N_dir-1) > Array_Grid_types {};
                                                             Array_Grid_types.at(0) = {'N','N'};
                                                             Array_Grid_types.at(1) = {'M','N'};
                                                             Array_Grid_types.at(2) = {'N','M'};
                                                             Array_Grid_types.at(3) = {'M','M'};
        return Array_Grid_types;
    }
    // NOTE: Without constexpr, the above function should lead to ODR error since this .hpp is 
    //       included (directly or indirectly) in different compilation units.
    //           If constexpr is evaluated at compile time, which is my use case, see following:
    //              constexpr auto Array_Grid_types = ns_forward::define_Array_Grid_types ();
    //       The constexpr specifier before auto should demand the function be evaluated at 
    //       compile time. (constexpr functions can be evaluated at compile or run time depending 
    //       on whether the variables it needs are available at compile time; however, definition
    //       of constexpr Array_Grid_types needs to be known at compile time, therefore, the 
    //       function define_Array_Grid_types needs to be evaluated at compile time.)
    //           I could use consteval to be more expressive of the restriction that this function
    //       should only be used at compile time (otherwise ODR). But consteval is a newer keyword,
    //       some compiler versions may not recognize it.
    //           If the function is evaluated at compile time, no run time object is generated for
    //       the function (in my case, the evaluated result is directly assigned to the variable), 
    //       therefore no ODR.


    // // static 
    // // inline
    // static inline auto define_Map_G_type_to_index ()
    // {
    //     std::map< std::array<char, N_dir> , int > Map_G_type_to_index {};
    //                                               Map_G_type_to_index[ {'N','N'} ] = 0;
    //                                               Map_G_type_to_index[ {'M','N'} ] = 1;
    //                                               Map_G_type_to_index[ {'N','M'} ] = 2;
    //                                               Map_G_type_to_index[ {'M','M'} ] = 3;
    //     return Map_G_type_to_index;
    // }
    // // NOTE: This function cannot be constexpr specified (yet, 2021) because std::map doesn't have
    // //       a constexpr implementation yet.
    // //           Without any specifier, ODR.
    // //           With static, ODR goes away, compiler warns "defined but not used" for every compilation
    // //       unit that (directly or indirectly) include this header file.
    // //           With inline, no ODR, no warning.
    // //           With static inline, no ODR, no warning.
    // //
    // //           static works because it restricts the scope of the defined function (it only has internal 
    // //       linkage, not visible outside); see the third usage from the first reference.
    // //           inline works because inlining replaces the function call with the function body, linking 
    // //       is not needed. This is similar to constexpr outlined above, but with a subtle difference - 
    // //       constexpr outlined above happens at compile time; actually, I should say constexpr is similar
    // //       to inline - in fact, if evaluated at run time, constexpr implies inlining.
    // //           static inline has even more nuances, see the second reference.
    // //  
    // //           Finally, this function is likely never to be called.
    // //
    // // References:
    // //     1) on the three usage of keyword static in C++ : http://www.mjbshaw.com/2012/11/the-static-modifier.html
    // //     2) https://stackoverflow.com/a/38043566
    // //     3) https://stackoverflow.com/a/56172321 (VERY well explained)
    // //     4) Nuance on C99 from clang implementation : https://clang.llvm.org/compatibility.html#inline

    // I still cannot constexpr std::string;

    // int test_int; // This would cause conflict (ODR) at linking time if this file is included twice;
                     // Well, now we know there are constexpr, static, inline, anonymous namespace ... 
}

namespace ns_forward = namespace_forward;

#endif