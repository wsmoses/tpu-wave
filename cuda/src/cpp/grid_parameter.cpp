#include "grid.hpp"

#include "inverse.hpp"
#include "namespace_input.hpp"

template<typename T>
void Class_Grid::interpolate_forward_parameter ( 
                                                 run_time_vector< T > & forward_P )
{
    using namespace ns_input;


} // interpolate_forward_parameter


// NOTE: We can fix the types of foward parameters, which are stored in grids. 
//       For example, we can make it always rho, lambda, and mu. 
//           The types of inverse parameters can be made free to change, which
//       are stored in Inv_Specs.
//           The following function would need to be modified if the inverse 
//       parameters are not rho, vp, and vs. For example, they can be rho, 
//       lambda and mu directly, or they can be log of the 'true' parameters.
//           Alternatively, we can convert the actual parameters used in
//       optimization to rho, vp, and vs first, i.e., using rho, vp, and vs 
//       as intermediate variables. The conversion would then be done in 
//       Inv_Specs. This would be confusing if the inverse parameters are rho, 
//       lambda, and mu or their log verions.
void Class_Grid::retrieve_forward_parameter ( Class_Inverse_Specs &Inv_Specs )
{

} // retrieve_forward_parameter

