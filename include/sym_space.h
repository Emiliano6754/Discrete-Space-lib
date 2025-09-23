#ifndef SYM_SPACE_H
#define SYM_SPACE_H

#include<Eigen/Dense>
#include<unsupported/Eigen/CXX11/Tensor>

// Executes func(int m, int n, int k) over the whole valid triples (m, n, k) of the symmetric space
template <typename Func>
void sym_space_loop(const unsigned int &n_qubits, Func func) {
    int k_max;
    for (int m = 0; m < n_qubits + 1; m++) {
        for (int n = 0; n < n_qubits + 1; n++) {
            k_max = std::min(m + n, 2*static_cast<int>(n_qubits) - m - n) + 1;
            for (int k = std::abs(m - n); k < k_max; k += 2) {
                func(m, n, k);
            }
        }
    }
}

// Returns a mask with 1s on valid triples (m, n, k) inside the symmetric space and 0s everywhere else
Eigen::Tensor<double, 3> sym_space_mask(const unsigned int &n_qubits);

// Returns a tensor filled with the values R_{m,n,k}
Eigen::Tensor<double, 3> get_Rmnk(const unsigned int &n_qubits);

// Returns the P function of S•v evaluated in (m, n, k), assuming v is normalized
inline double Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v, const unsigned int &m, const unsigned int &n, const unsigned int &k) {
    return std::sqrt(3.0) * (n_qubits - 2.0 * (m * v(0) + n * v(1) + k * v(2) ) ) / qubitstate_size;
}

// Returns the P function of S•v, evaluated in (m, n, k),  where v is defined by the unit vector v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
inline double Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, const unsigned int &m, const unsigned int &n, const unsigned int &k) {
    return std::sqrt(3.0) * (n_qubits - 2.0 * (m * std::sin(theta) * std::cos(phi) + n * std::sin(theta) * std::sin(phi) + k * std::cos(theta) ) ) / qubitstate_size;
}

// Returns a tensor with the P function of S•v, assuming v is normalized
Eigen::Tensor<double, 3> get_Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v);

// Returns the P function of S•v, evaluated in (m, n, k),  where v is defined by the unit vector v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
inline double Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, const unsigned int &m, const unsigned int &n, const unsigned int &k);

// Returns a tensor with the P function of S•v, assuming v is normalized,  where v is defined by the unit vector v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
Eigen::Tensor<double, 3> get_Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi);

// Returns the P function of Sx/Sy/Sz. They are all equal, with the only difference being which variable is spanned by the single dimension. Notice that on broadcasting only valid triples (m, n, k) should be distinct from zero. If this is used to calculate averages, it is enough if the state sym Q is zero in those places
Eigen::Tensor<double, 1> get_cartesian_S_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

// Returns the P function of Sx^2/S_y^2/S_z^2. They are all equal, with the only difference being which variable is spanned by the single dimension. Notice that on broadcasting only valid triples (m, n, k) should be distinct from zero. If this is used to calculate averages, it is enough if the state sym Q is zero in those places
Eigen::Tensor<double, 1> get_cartesian_S2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

// Returns the full symmetric P function of Sx
Eigen::Tensor<double, 3> get_Sx_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

// Returns the full symmetric P function of Sy
Eigen::Tensor<double, 3> get_Sy_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

// Returns the full symmetric P function of Sz
Eigen::Tensor<double, 3> get_Sz_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

// Returns the full symmetric P function of {Sy,Sz}
Eigen::Tensor<double, 3> get_aSySz_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

// Returns the full symmetric P function of {Sz,Sx}
Eigen::Tensor<double, 3> get_aSzSx_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

// Returns the full symmetric P function of {Sx,Sy}
Eigen::Tensor<double, 3> get_aSxSy_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size);

// Calculates the average of a symmetric operator. Accepts the operator_symP as a template to allow for tensor expressions to be passed
template <typename TensorExpr>
void sym_operator_average(const TensorExpr &operator_symP, const Eigen::Tensor<double, 3> &state_symQ, double &average);

// Calculates the expected value of S•v, where v is given as an Eigen::Vector3d and is assummed to be normalized
void ang_operator_average(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v, const Eigen::Tensor<double, 3> &state_symQ, double &average);

// Calculates the expected value of S•n, where n is given by its angles
void ang_operator_average(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, const Eigen::Tensor<double, 3> &state_symQ, double &average);

// Calculates the averages of all quadratic operators, where the cross products are replaced by the anticommutator. Can be optimized by reducing the size of Sx/Sy/Sz to their respective variables
void cartesian_ang_operator_averages(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &state_symQ, double &Sx, double &Sy, double &Sz, double &Sx2, double &Sy2, double &Sz2, double &SySz, double &SzSx, double &SxSy);

// Returns the correlation matrix for a particular symmetric Q function. In order to calculate values of the Gaussian envelope, the average values of Sx, Sy, Sz are returned in their inputs as well
Eigen::Matrix3d get_correlation_matrix(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ, double &Sx, double &Sy, double &Sz);

// Returns the Gaussian envelope of the state SymQ
Eigen::Tensor<double, 3> get_Gfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ);

// Returns the Gaussian envelope of the state SymQ in Gfunc to avoid copying
void get_Gfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ, Eigen::Tensor<double, 3> &Gfunc);

#endif