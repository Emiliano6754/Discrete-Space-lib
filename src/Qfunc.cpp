#include "Qfunc.h"

// Stores all powers of xi from 0 t n_qubits in xi_buffer. Assumes xi_buffer already allocates enough memory
inline void initialize_xi_buffer(const unsigned int &n_qubits, const std::complex<double> &xi, std::complex<double>* xi_buffer) {
    xi_buffer[0] = 1;
    for (unsigned int n = 1; n <= n_qubits; n++) {
        xi_buffer[n] = xi * xi_buffer[n - 1];
    }
} 

// Calculates the discrete Q function of a pure state from its operational basis expansion
Eigen::MatrixXd pure_Qfunc_from_operational(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::VectorXd &state) {
    Eigen::MatrixXd Qfunc(qubitstate_size, qubitstate_size);
    const std::complex<double> xi = 0.5 * (sqrt(3)-1) * std::complex<double>(1.0,1.0);
    const double norm = 1.0 / std::pow((1 + std::norm(xi)), n_qubits);
    std::complex<double>* xi_conj_buffer = static_cast<std::complex<double>*>( _malloca((n_qubits + 1) * sizeof(std::complex<double>)) );
    initialize_xi_buffer(n_qubits, std::conj(xi), xi_conj_buffer);
    
    #pragma omp parallel
    {
        std::complex<double> sum;

        #pragma omp for collapse(2)
        for (unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
            for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
                sum = 0;
                for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
                    sum += (1.0 - 2 * std::popcount(alpha & eta)) * xi_conj_buffer[std::popcount(beta ^ eta)] * state[eta];
                }
                Qfunc(alpha, beta) = norm * std::norm(sum);
            }
        }
    }
    return Qfunc;
}