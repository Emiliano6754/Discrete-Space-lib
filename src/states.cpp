#include "states.h"
#include<bit>
#include<complex>
#include<unsupported/Eigen/KroneckerProduct>

// Calculates binom(n_qubits, k)
double binom_coeff(const unsigned int &n_qubits, const unsigned int &k) {
    double binom = 1;
    for (int j = 0; j < k; j++) {
        binom *= static_cast<double>(n_qubits - j) / (k - j);
    }
    return binom;
}

// Returns the normalized state K(|N/2, N> + F|N/2, N>) where N = n_qubits. Requires n_qubits to be even.
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

// Returns the normalized Dicke state |k, n_qubits>.
Eigen::VectorXd Dicke_state(const unsigned int &n_qubits, const unsigned int &k, const unsigned int &qubitstate_size) {
    Eigen::VectorXd state(qubitstate_size);
    const double norm = 1.0 / std::sqrt(binom_coeff(n_qubits, k));
    for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
        state(eta) = (std::popcount(eta) == k);
    }
    state *= norm;
    return state;
}

// Returns the normalized GHZ state 1/sqrt(2)(|0,N> + |N,N>) with n_qubits
Eigen::VectorXd GHZ_state(const unsigned int &n_qubits) {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(1 << n_qubits);
    state(0) = 1.0 / std::sqrt(2);
    state((1 << n_qubits) - 1) = 1.0 / std::sqrt(2);
    return state;
}

// Returns the normalized cluster state with n_qubits
Eigen::VectorXd cluster_state(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(qubitstate_size);
    const double norm = 1.0 / std::sqrt(1 << n_qubits);
    unsigned int upper_bit_mask = 1 << (n_qubits - 1); 
    int sign_sum = 0;
    for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
        sign_sum = std::popcount(eta & (eta << 1)) + (eta & upper_bit_mask) ? eta & 1 : 0;
        state(eta) = norm * ( 1.0 - 2 * (sign_sum % 2) );
    }
    return state;
}

// Returns the normalized singlet state (n_qubits/2 pairs of bi-partite singlet states) with n_qubits even
Eigen::VectorXd singlet_state(const unsigned int &n_qubits) {
    Eigen::Vector4d singlet_pair{0, 1.0 / std::sqrt(2), 0, -1.0 / std::sqrt(2)};
    Eigen::VectorXd state = singlet_pair;
    const unsigned int n_pairs = n_qubits / 2;
    Eigen::MatrixXd product;
    for (unsigned int j = 1; j < n_pairs; j++) {
        product = Eigen::kroneckerProduct(singlet_pair, state);
        state = product.reshaped();
    }
    return state;
}

// Returns the normalized SU(2) coherent state with n_qubits, parametrized by the angles theta, phi
Eigen::VectorXcd su2_coherent_state(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const float &theta, const float &phi) {
    Eigen::VectorXcd state(qubitstate_size);
    const std::complex<double> zero_coeff(std::cos(theta / 2), 0);
    const std::complex<double> one_coeff = std::sin(theta / 2) * std::complex(std::cos(phi), std::sin(phi));
    for (unsigned int eta = 0; eta < qubitstate_size; eta++) {
        state(eta) = std::pow(one_coeff, std::popcount(eta)) * std::pow(zero_coeff, n_qubits - std::popcount(eta));
    }
    return state;
}

// Returns the normalized W state with n_qubits
Eigen::VectorXd W_state(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(qubitstate_size);
    const double norm = 1.0 / std::sqrt(n_qubits);
    for (int j = 0; j < n_qubits; j++) {
        state(1 << j) = norm;
    }
    return state;
}