#ifndef STATESH
#define STATESH

#include<Eigen/Dense>

// Returns the normalized state K(|N,N/2> + F|N,N/2>) where N = n_qubits. Requires n_qubits to be even.
Eigen::VectorXd Dicke_superposition(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

#endif