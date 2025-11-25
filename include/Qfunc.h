#ifndef QFUNC
#define QFUNC

#include<Eigen/Dense>
#include<complex>

// Calculates the discrete Q function of a pure state from its operational basis expansion, assuming the state has no imaginary part
Eigen::MatrixXd pure_Qfunc_from_operational(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::VectorXd &state);

// Calculates the discrete Q function of a pure state from its operational basis expansion
Eigen::MatrixXd pure_Qfunc_from_operational(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::VectorXcd &state);

// Prints the Qfunc to console as a function of alpha and beta for debugging purposes 
void print_Qfunc(const Eigen::MatrixXd &Qfunc);

// Saves the Qfunc to filename in numpy style
void save_Qfunc(const Eigen::MatrixXd &Qfunc, const std::string &filename);

#endif