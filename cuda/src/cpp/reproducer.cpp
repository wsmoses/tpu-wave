#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <math.h>

// Mock ns_input variables
namespace ns_input {
    int G_size_x = 601;
    int G_size_y = 601;
    char G_type_x = 'N';
    char G_type_y = 'N';
    int num_dx_soln = 1;
    int den_dx_soln = 1;
    int num_dx_prmt = 1;
    int den_dx_prmt = 1;
    int Nx_prmt = 601;
    int Ny_prmt = 601;
}

template<typename T>
class run_time_vector {
public:
    int length = 0;
    T* ptr = nullptr;

    run_time_vector(int len) : length(len) {
        ptr = (T*)malloc(len * sizeof(T));
        memset(ptr, 0, len * sizeof(T));
    }

    run_time_vector(const run_time_vector& v) : length(v.length) {
        ptr = (T*)malloc(length * sizeof(T));
        for (int i = 0; i < length; i++) {
            ptr[i] = v.ptr[i];
        }
    }

    ~run_time_vector() {
        free(ptr);
    }

    T& at(int i) {
        if (i < 0 || i >= length) {
            printf("Out of bound access - at - run_time_vector. i=%d, length=%d\n", i, length);
            exit(0);
        }
        return ptr[i];
    }

    T& operator()(int i) {
        return at(i);
    }
};

class Grid {
public:
    std::vector<run_time_vector<float>> Vec_prmt;

    Grid(int length) {
        Vec_prmt.push_back(run_time_vector<float>(length));
    }

    template<typename T>
    void interpolate_forward_parameter(run_time_vector<double>& inverse_P,
                                       run_time_vector<T>& forward_P) {
        using namespace ns_input;

        for (int ix_S = 0; ix_S < G_size_x; ix_S++) {
            for (int iy_S = 0; iy_S < G_size_y; iy_S++) {
                long num_x_loc = 1 << 30; long den_x_loc = -1;
                long num_y_loc = 1 << 30; long den_y_loc = -1;

                if (G_type_x == 'N') { num_x_loc = ix_S * num_dx_soln; den_x_loc = den_dx_soln; }
                if (G_type_y == 'N') { num_y_loc = iy_S * num_dx_soln; den_y_loc = den_dx_soln; }

                if (G_type_x == 'M') { num_x_loc = (2 * ix_S + 1) * num_dx_soln; den_x_loc = 2 * den_dx_soln; }
                if (G_type_y == 'M') { num_y_loc = (2 * iy_S + 1) * num_dx_soln; den_y_loc = 2 * den_dx_soln; }

                long ix_M = (num_x_loc * den_dx_prmt) / (den_x_loc * num_dx_prmt);
                long iy_M = (num_y_loc * den_dx_prmt) / (den_y_loc * num_dx_prmt);

                if (ix_M >= Nx_prmt) ix_M = Nx_prmt - 1;
                if (iy_M >= Ny_prmt) iy_M = Ny_prmt - 1;
                if (ix_M < 0) ix_M = 0;
                if (iy_M < 0) iy_M = 0;

                double xi = static_cast<double>(num_x_loc * den_dx_prmt) / static_cast<double>(den_x_loc * num_dx_prmt) - ix_M;
                double eta = static_cast<double>(num_y_loc * den_dx_prmt) / static_cast<double>(den_y_loc * num_dx_prmt) - iy_M;

                const int PADDED_stride = Ny_prmt + 1;

                double v = (1 - xi) * (1 - eta) * inverse_P((ix_M) * PADDED_stride + (iy_M))
                    + (1 - xi) * (eta)*inverse_P((ix_M) * PADDED_stride + (iy_M + 1))
                    + (xi) * (1 - eta) * inverse_P((ix_M + 1) * PADDED_stride + (iy_M))
                    + (xi) * (eta)*inverse_P((ix_M + 1) * PADDED_stride + (iy_M + 1));

                forward_P(ix_S * G_size_y + iy_S) = v;
            }
        }
    }

    void make_density_reciprocal() {
        run_time_vector<float>& P = Vec_prmt.at(0);
        run_time_vector<double> forward_P(P.length);
        
        // inverse_P needs to be large enough for interpolation
        run_time_vector<double> inverse_P((ns_input::Nx_prmt + 2) * (ns_input::Ny_prmt + 2));
        for(int i=0; i<inverse_P.length; i++) inverse_P.at(i) = 2.0; // fill with something

        interpolate_forward_parameter<double>(inverse_P, forward_P);

        std::cout << "Starting loop..." << std::endl;
        for (int i = 0; i < P.length; i++) {
            if (i == 359999) {
                std::cout << "Reached 359999" << std::endl;
            }
            P.at(i) = 1.0 / forward_P.at(i);
        }
        std::cout << "Success!" << std::endl;
    }
};

int main() {
    Grid g(361201);
    g.make_density_reciprocal();
    return 0;
}
