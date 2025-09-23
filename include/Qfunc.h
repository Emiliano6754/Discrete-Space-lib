#ifndef QFUNC
#define QFUNC

#include<Eigen/Dense>
#include<complex>

// Calculates the discrete Q function of a pure state from its operational basis expansion
Eigen::MatrixXd pure_Qfunc_from_operational(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::VectorXd &state);

#endif