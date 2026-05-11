#ifndef NAMESPACE_TYPE_HPP
#define NAMESPACE_TYPE_HPP

#include <string>
#include <typeinfo>
#include <cstdio>

namespace namespace_type
{
    using cuda_precision = double;
    using host_precision = cuda_precision;
}
namespace ns_type = namespace_type;

inline void print_precision_type ()
{
    std::string str_precision = "double";
    printf( "\nPrecision type: %s .\n", str_precision.c_str() );
}

#endif
