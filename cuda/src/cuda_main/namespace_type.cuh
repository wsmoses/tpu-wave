#ifndef NAMESPACE_TYPE_NVCC
#define NAMESPACE_TYPE_NVCC

// #include "cuda_fp16.h"
// #include "cuda_bf16.h"

// #include <string>
// #include <typeinfo>

namespace namespace_type
{
#ifndef PRCSFLAG
    using cuda_precision = double;
#endif

#if PRCSFLAG == 64
    using cuda_precision = double;
#elif PRCSFLAG == 32    
    using cuda_precision = float;
#elif PRCSFLAG == 16        
    using cuda_precision = __half;
#endif

    // using cuda_precision = __float128;
    // using cuda_precision = long double;
    // using cuda_precision = double;
    // using cuda_precision = float;
    // using cuda_precision = __half;
    // using cuda_precision = __nv_bfloat16;
    // using cuda_precision = _Float16;

    using host_precision = cuda_precision;
}
namespace ns_type = namespace_type;


inline void print_precision_type ()
{
    std::string str_precision = "Unrecognized type";
    if ( typeid(ns_type::cuda_precision) == typeid(double)        ) { str_precision = "double";        } 
    if ( typeid(ns_type::cuda_precision) == typeid(float)         ) { str_precision = "float" ;        } 
    // if ( typeid(ns_type::cuda_precision) == typeid(__half)        ) { str_precision = "__half";        } 
  //  if ( typeid(ns_type::cuda_precision) == typeid(__nv_bfloat16) ) { str_precision = "__nv_bfloat16"; }
    if ( typeid(ns_type::cuda_precision) == typeid(_Float16)      ) { str_precision = "_Float16";      }  

    printf( "\nPrecision type: %s .\n", str_precision.c_str() );
}

// [2023/09/20]
// NOTE: we should change the above usage to is_same_v. The comparison can 
//       be done at compile time.


#endif

// [2023/06/10]
// NOTE: This file has the same name as the one in ../cpp_main so that we 
//       could #include "namespace_type.cuh" in namespace_type.hpp without 
//       change for both gcc compilation (with gcc type _Float16) and nvcc
//       compilation (with nvcc types __half and __nv_bfloat16). 
//       
//       The header guards are different so that if both files are included 
//       (unintentionally), there "may be" be a conflict in defining ns_type 
//       twice (if both files are included in the same translation unit).
