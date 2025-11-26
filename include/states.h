#ifndef STATESH
#define STATESH

#include<Eigen/Dense>

// Returns the normalized state K(|N,N/2> + F|N,N/2>) where N = n_qubits. Requires n_qubits to be even.
Eigen::VectorXd Dicke_superposition(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

// Returns the normalized Dicke state |n_qubits,k>.
Eigen::VectorXd Dicke_state(const unsigned int &n_qubits, const unsigned int &k, const unsigned int &qubitstate_size);

// Returns the normalized GHZ state 1/sqrt(2)(|0,N> + |N,N>) with n_qubits
Eigen::VectorXd GHZ_state(const unsigned int &n_qubits);

// Returns the normalized cluster state with n_qubits
Eigen::VectorXd cluster_state(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

// Returns the normalized singlet state (n_qubits/2 pairs of bi-partite singlet states) with n_qubits even
Eigen::VectorXd singlet_state(const unsigned int &n_qubits);

// Returns the normalized SU(2) coherent state with n_qubits, parametrized by the angles theta, phi
Eigen::VectorXcd su2_coherent_state(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const float &theta, const float &phi);

// Returns the normalized domain wall state with n_qubits
Eigen::VectorXd domain_wall_state(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

#endif