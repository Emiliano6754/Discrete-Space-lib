#include "states.h"

// Calculates binom(n_qubits, k)
double binom_coeff(const unsigned int &n_qubits, const unsigned int &k) {
    double binom = 1;
    for (int j = 0; j < k; j++) {
        binom *= static_cast<double>(n_qubits - j) / (k - j);
    }
    return binom;
}

// Returns the normalized state K(|N,N/2> + F|N,N/2>) where N = n_qubits. Requires n_qubits to be even.
Eigen::VectorXd Dicke_superposition(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::VectorXd state(qubitstate_size);
    const unsigned int half = n_qubits/2;
    const double coef = 1.0 / (1 << half);
    int sum = 0;
    for (unsigned int k = 0; k < qubitstate_size; k++) {
        sum = 0;
        for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
            sum += (std::popcount(eta) == half) * (1.0 - 2 * std::popcount(eta & k));
        }
        state(k) = (std::popcount(k) == half) + coef * sum;
    }
    state.normalize();
    return state;
}

// Returns the normalized state Dicke state |n_qubits,k>.
Eigen::VectorXd Dicke_state(const unsigned int &n_qubits, const unsigned int &k, const unsigned int &qubitstate_size) {
    Eigen::VectorXd state(qubitstate_size);
    const double norm = 1.0 / std::sqrt(binom_coeff(n_qubits, k));
    for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
        state(eta) = (std::popcount(eta) == k);
    }
    state.normalize();
    return state;
}