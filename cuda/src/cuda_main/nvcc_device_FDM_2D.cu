
#include <stdio.h>
#include <array>
#include <map>
#include <vector>
    // constexpr auto define_Array_Grid_types ()
    consteval auto define_Array_Grid_types ()    // if some compiler versions complain,
    {                                            // try changing consteval to constexpr
        std::array< std::array<char, 2>, 4 > Array_Grid_types {};
                                                             Array_Grid_types.at(0) = {'N','N'};
                                                             Array_Grid_types.at(1) = {'M','N'};
                                                             Array_Grid_types.at(2) = {'N','M'};
                                                             Array_Grid_types.at(3) = {'M','M'};
        return Array_Grid_types;
    }

int main(int argc, char* argv[]) 
{
   
           std::map< std::array<char,2> , std::vector<double> > Map_grid_record_rcv;  // involve memory

    constexpr auto Array_Grid_types = define_Array_Grid_types ();
    for (int i=0; i<1; i++) {

        for ( const auto & grid_type : Array_Grid_types ) { 
		Map_grid_record_rcv        [grid_type] = {};
	  printf("saw grid size %c %c\n", grid_type[0], grid_type[1]);
	}

    }

    return 0;
}
