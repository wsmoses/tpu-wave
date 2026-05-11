#include <iostream>
#include <vector>
#include <map>
#include <array>

struct struct_rcv_input {
    std::array<char, 2> rcv_grid_type;
    int rcv_index;
};

int main() {
    std::array<std::array<char, 2>, 4> Array_Grid_types = {
        std::array<char, 2>{'N', 'N'},
        std::array<char, 2>{'M', 'N'},
        std::array<char, 2>{'N', 'M'},
        std::array<char, 2>{'M', 'M'}
    };

    std::map<std::array<char, 2>, int> Map_grid_N_rcvs;
    std::map<std::array<char, 2>, std::vector<int>> Map_grid_record_rcv;

    for (const auto& grid_type : Array_Grid_types) {
        Map_grid_N_rcvs[grid_type] = 0;
        Map_grid_record_rcv[grid_type] = {};
    }

    std::vector<struct_rcv_input> Vec_Rcv_Input;
    Vec_Rcv_Input.push_back({{'N', 'M'}, 1});

    std::cout << "Starting loop..." << std::endl;

    for (const auto& rcv_input : Vec_Rcv_Input) {
        std::array<char, 2> grid_type = rcv_input.rcv_grid_type;

        if (true) {
            Map_grid_N_rcvs.at(grid_type) += 1;
            Map_grid_record_rcv.at(grid_type).push_back(1);
        }
    }

    std::cout << "Success!" << std::endl;
    return 0;
}
