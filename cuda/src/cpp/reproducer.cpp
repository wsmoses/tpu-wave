#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>

template<typename T>
class run_time_vector {
public:
    int length = 0;
    T* ptr = nullptr;

    run_time_vector(int len) : length(len) {
        ptr = (T*)malloc(len * sizeof(T));
        memset(ptr, 0, len * sizeof(T));
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
};

class Grid {
public:
    std::vector<run_time_vector<double>> Vec_prmt;

    Grid(int length) {
        Vec_prmt.push_back(run_time_vector<double>(length));
    }

    void make_density_reciprocal() {
        run_time_vector<double>& P = Vec_prmt.at(0);
        run_time_vector<double> forward_P(P.length);

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
