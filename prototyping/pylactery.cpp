// phylactery.cpp - Compile with: g++ -O3 -std=c++17 -shared -fPIC -o phylactery.so phylactery.cpp

#include <cmath>
#include <complex>
extern "C" {
    double iterate(double real, double imag, int max_iter, double escape_radius) {
        std::complex<double> z(0.0, 0.0);
        std::complex<double> c(real, imag);
        double escape_radius_sq = escape_radius * escape_radius;

        for (int i = 0; i < max_iter; ++i) {
            // A bizarre custom recursive rule approximating your original recurrence
            z = z*z + c + 0.25 / (std::abs(z) + 1e-9); // add a nonlinearity

            if (std::norm(z) > escape_radius_sq) {
                return static_cast<double>(i);
            }
        }
        return static_cast<double>(max_iter);
    }
}