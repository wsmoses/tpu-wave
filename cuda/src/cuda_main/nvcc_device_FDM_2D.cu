#include "grid.cuh"
int main(int argc, char* argv[]) 
{

    int prmt_M_sizes[2] = {600, 600};
    int soln_M_sizes[2] = {600, 600};
    int num_dx_prmt = 8;
    int den_dx_prmt = 1000;
    int num_dx_soln = 8;
    int den_dx_soln = 1000;
    int Nt = 60000;
    double CFL_constant = 0.625;
    char c_energy = 'Y';
    bool bool_energy = true;

    double dx = static_cast<double>( num_dx_soln ) / static_cast<double>( den_dx_soln );
    double dt = 0.0001 * CFL_constant; // Assuming dt_max was 0.0001 based on previous logs

    cuda_Class_Grid<'N', 'N', 601, 601> grid_SNN;
    cuda_Class_Grid<'M', 'M', 600, 600> grid_SMM;

    grid_SMM.cuda_Class_Grid_initialize ( "SMM", 2, 2 );
    grid_SNN.cuda_Class_Grid_initialize ( "Sxy", 1, 1 );
{

    for (int it=0; it<Nt; it++) 
    {


		grid_SMM.energy_calculation ();
		grid_SNN.energy_calculation ();

    }  // for (int it=0; it<Nt; it++) 


} 

    return 0;
}
