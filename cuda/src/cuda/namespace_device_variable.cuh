#ifndef NAMESPACE_DEVICE_VARIABLE_H
#define NAMESPACE_DEVICE_VARIABLE_H

#include "namespace_type.cuh"

namespace namespace_device_variable
{
    // __device__ inline ns_type::cuda_precision dx;
    // __device__ inline ns_type::cuda_precision dt;
}
namespace ns_dev_var = namespace_device_variable;

// static __global__ void cuda_print_dt ()
//     { printf( " %16.15e \n", static_cast<double> ( ns_dev_var::dt ) ); }
// [2023/07/16]
// NOTE: "static" means every file that includes this header gets a private
//       (i.e., internal linkage, i.e., only visible in this TU) copy of the 
//       above function. This is to avoid the "multiple definition error" at 
//       the linking stage. 
//           /* Using the inline keyword on __global__ functions leads to 
//              warning at least. */

template<typename T>
__global__ void cuda_print_test ( T * v )
    { printf( " %16.15e \n", static_cast<double> ( *v ) ); }

#endif