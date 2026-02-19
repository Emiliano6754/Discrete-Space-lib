#include "sym_space.h"
#include<cmath>
#include<complex>
#include<bit>
#include<fstream>
#include<iostream>
#include<filesystem>
#include<vector>
#include<Qfunc.h>

static constexpr double sqrt3 = 1.73205080756887;
static constexpr std::complex<double> xi = std::complex<double>(0.5 * (sqrt3 - 1), 0.5 * (sqrt3 - 1));
static constexpr std::complex<double> xi_conj = std::complex<double>(0.5 * (sqrt3 - 1), -0.5 * (sqrt3 - 1));
static constexpr double xi_sum_inv = sqrt3; // (1 + abs(xi)^2)/(xi + xi_conj)
static constexpr std::complex<double> xi_min_inv = std::complex<double>(0.0, sqrt3); // (1 + abs(xi)^2)/(xi_conj - xi)
static constexpr double xi_norm_inv = sqrt3; // (1 + abs(xi)^2)/(1 - abs(xi)^2)

static unsigned int fact(const unsigned int &n) {
    if (n == 0) { 
        return 1;
    }
    unsigned int res = n;
    for (unsigned int j = 2; j < n; j++) {
        res *= j;
    }
    return res;
}

// Returns Binom(N,k) exactly
static unsigned int binom(const unsigned int &N, const unsigned int &k) {
    unsigned int res = 1;
    for (unsigned int j = N; j > N - k; j--) {
        res *= j;
    }
    res /= fact(k);
    return res;
}

// Returns a double approximation of Binom(N,k)
static double double_binom(const unsigned int &N, const unsigned int &k) {
    double res = 1;
    for (int j = 1; j <= k; j++) {
        res *= static_cast<double>(N + 1 - j) / j;
    }
    return res;
}

// Returns a tensor of all double approximations of Binom(N,k) from k=0 to k=N
static std::vector<double> double_binoms(const unsigned int &N) {
    std::vector<double> res(N+1);
    for (unsigned int k = 0; k < N+1; k++) {
        res[k] = double_binom(N, k);
    }
    return res;
}

// Returns a tensor of doubles filled with all binomials (N,k) from k=0 to k=N
static Eigen::Tensor<double, 1> binom(const unsigned int &N) {
    Eigen::Tensor<double, 1> res(N+1);
    for (unsigned int k = 0; k < N+1; k++) {
        res(k) = binom(N, k);
    }
    return res;
}

template <typename... Ints>
static Eigen::array<unsigned int, sizeof...(Ints)> ind_arr(Ints... index) {
    return {static_cast<unsigned int>(index)...};
}

// Returns a mask with 1s on valid triples (m, n, k) inside the symmetric space and 0s everywhere else
Eigen::Tensor<double, 3> sym_space_mask(const unsigned int &n_qubits) {
    Eigen::Tensor<double, 3> mask(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    mask.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        mask(m, n, k) = 1;
    });
    return mask;
}

// Returns the value of R_{m,n,k}
inline double Rmnk(const unsigned int &n_qubits, const unsigned int &m, const unsigned int &n, const unsigned int &k) {
    return double_binom(n_qubits, m) * double_binom(m, (m+n-k)/2) * double_binom(n_qubits - m, (-m + n + k)/2);
}

// Returns a tensor filled with the values R_{m,n,k}
Eigen::Tensor<double, 3> get_Rmnk(const unsigned int &n_qubits) {
    Eigen::Tensor<double, 3> R(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    R.setZero();
    sym_space_loop(n_qubits, [&](int const &m, int const &n, int const &k) {
        R(m, n, k) = Rmnk(n_qubits, m, n, k);
    });
    return R;
}

// Returns the number of combinations of 3 field variables, with fixed weights h. This is exact, but overflows quickly
unsigned int C3(const unsigned int &n_qubits, const int &h1, const int &h2, const int &h3, const int &h4, const int &h5, const int &h6, const int &h7) {
    const int w1 = (-h1 - h2 - h3 + h4 + h5 + h6 + h7) / 4;
    const int w2 = (h1 - h2 + h3 + h4 - h5 + h6 - h7) / 4;
    const int w3 = (-h1 + h2 + h3 + h4 + h5 - h6 - h7) / 4;
    const int w4 = (h1 + h2 - h3 + h4 - h5 - h6 + h7) / 4;
    const int N1 = n_qubits - (h1 + h2 + h3) / 2;
    const int N2 = (h1 - h2 + h3) / 2;
    const int N3 = (-h1 + h2 + h3) / 2;
    const int N4 = (h1 + h2 - h3) / 2;
    if (w1 < 0 || w2 < 0 || w3 < 0 || w4 < 0 || N1 - w1 < 0 || N2 - w2 < 0 || N3 - w3 < 0 || N4 - w4 < 0) {
        return 0;
    }
    return fact(n_qubits) / (fact(w1) * fact(w2) * fact(w3) * fact(w4) * fact(N1 - w1) * fact(N2 - w2) * fact(N3 - w3) * fact(N4 - w4));
}

// Checks that the weights h1, h2, h3 can be obtained from a relation akin to h3 = h1 + h2. Returns true if they can't
static bool check_pairwise_weights(const unsigned int &n_qubits, const int &h1, const int &h2, const int &h3) {
    return (h3 < std::abs(h1 - h2) || h3 > std::min(h1 + h2, static_cast<int>(2 * n_qubits) - h1 - h2));
}

// Returns sum_h7 R_{h1h2h3}^{-1} C3(h). Checked for all possibilities with n_qubits = 3 and they are correctly generated. Could compare with a direct delta sum for bigger n_qubits
double reduced_C3(const unsigned int &n_qubits, const int &h1, const int &h2, const int &h3, const int &h4, const int &h5, const int &h6) {
    if (check_pairwise_weights(n_qubits, h1, h2, h3) || check_pairwise_weights(n_qubits, h1, h4, h5) || check_pairwise_weights(n_qubits, h2, h4, h6)) {
        return 0;
    }
    double res = 0;
    double temp = 0;
    int c1, c2, c3, c4, w1, w2, w3, w4, N1, N2, N3, N4;
    c1 = (h1 + h2 + h3);
    c2 = (h1 - h2 + h3);
    c3 = (-h1 + h2 + h3);
    c4 = (h1 + h2 - h3);
    if (c1 % 2 || c2 % 2 || c3 % 2 || c4 % 2) {
        return 0;
    }
    N1 = n_qubits - c1 / 2;
    N2 = c2 / 2;
    N3 = c3 / 2;
    N4 = c4 / 2;
    for (int h7 = 0; h7 <= n_qubits; h7++) {
        if (check_pairwise_weights(n_qubits, h1, h6, h7) || check_pairwise_weights(n_qubits, h2, h5, h7) || check_pairwise_weights(n_qubits, h3, h4, h7)) {
            continue;
        }
        w1 = (-h1 - h2 - h3 + h4 + h5 + h6 + h7);
        w2 = (h1 - h2 + h3 + h4 - h5 + h6 - h7);
        w3 = (-h1 + h2 + h3 + h4 + h5 - h6 - h7);
        w4 = (h1 + h2 - h3 + h4 - h5 - h6 + h7);
        if (w1 < 0 || w1 % 4 || w2 < 0 || w2 % 4 || w3 < 0 || w3 % 4 || w4 < 0 || w4 % 4) {
            continue;
        }
        temp = double_binom(N1, w1/4) * double_binom(N2, w2/4) * double_binom(N3, w3/4) * double_binom(N4, w4/4);
        res += temp;
        std::cout << "Full C_3(" << h1 << ", " << h2 << ", " << h3 << ", " << h4 << ", " << h5 << ", " << h6 << ", " << h7 << ") = " << temp << std::endl;
    }
    return res;
}

// Returns renormalized g_mnk(p,q,r) as a kravchuk_exp on m, n, k, with p, q, r as parameters. To get g_mnk, evaluate gmnk with binom_eval or output as tensor with as_binom_tensor. If p, q, r are not valid weights, returns a zero expansion
kravchuk_exp get_kravchuk_gmnk(unsigned int const &n_qubits, unsigned int const &p, unsigned int const &q, unsigned int const &r) {
    kravchuk_exp gmnk(n_qubits);
    // Check if the parameters are valid weights
    if (r < std::abs(static_cast<int>(p) - static_cast<int>(q)) || r > std::min(static_cast<int>(p + q), 2*static_cast<int>(n_qubits) - static_cast<int>(p + q))) {
        return gmnk;
    }
    std::vector<double> Nchoose = double_binoms(n_qubits);
    double norm = 1.0 / (1 << n_qubits);
    // The limits of the inner sums can be reduced, as j1 forces all possible values of j2 and j3 to stay within some bounds, with p, q, r fixed. Bug in tensor product constructor
    for (int j3 = 0; j3 <= n_qubits; j3++) {
        for (int j2 = 0; j2 <= n_qubits; j2++) {
            for (int j1 = 0; j1 <= n_qubits; j1++) {
                gmnk(j1, j2, j3) = norm * reduced_C3(n_qubits, p, q, r, j1, j2, j3) / (Nchoose[j1] * Nchoose[j2] * Nchoose[j3]);
            }
        }
    }
    return gmnk;
}

// Returns g_mnk(p,q,r) evaluated over all symmetric space, with p, q, r as parameters. If p, q, r are not valid weights, returns a zero tensor
Eigen::Tensor<double, 3> get_gmnk(unsigned int const &n_qubits, unsigned int const &p, unsigned int const &q, unsigned int const &r) {
    // Check if the parameters are valid weights
    if (r < std::abs(static_cast<int>(p) - static_cast<int>(q)) || r > std::min(static_cast<int>(p + q), 2*static_cast<int>(n_qubits) - static_cast<int>(p + q))) {
        return Eigen::Tensor<double, 3>(n_qubits + 1, n_qubits + 1, n_qubits + 1).setZero();
    }
    kravchuk_exp kravchuk_gmnk = get_kravchuk_gmnk(n_qubits, p, q, r);
    return kravchuk_gmnk.as_binom_tensor();
}

// Returns g_mnk(p,q,r) for (p,q,r) in all symmetric space, stored in the order p, q, r
std::vector<kravchuk_exp> get_all_gmnk(unsigned int const &n_qubits) {
    std::cout << "Calculating all gmnk" << std::endl;
    std::vector<kravchuk_exp> gmnk;
    gmnk.reserve((n_qubits + 1) * (n_qubits + 1) * (n_qubits + 1));
    for (int r = 0; r <= n_qubits; r++) {
        for (int q = 0; q <= n_qubits; q++) {
            for (int p = 0; p <= n_qubits; p++) {
                gmnk.emplace_back(get_kravchuk_gmnk(n_qubits, p, q, r));
            }
        }
    }
    std::cout << "Finished calculating all gmnk" << std::endl;
    return gmnk;
}

// Returns a tensor with the P function of S•v, assuming v is normalized
Eigen::Tensor<double, 3> get_Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = Sv_Pfunc(n_qubits, qubitstate_size, v, m, n, k);
    });
    return P;
}

// Returns a tensor with the P function of S•v, assuming v is normalized,  where v is defined by the unit vector v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
Eigen::Tensor<double, 3> get_Sv_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi) {
    return get_Sv_Pfunc(n_qubits, qubitstate_size, {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)});
}

// Returns the P function of Sx/Sy/Sz. They are all equal, with the only difference being which variable is spanned by the single dimension. Notice that on broadcasting only valid triples (m, n, k) should be distinct from zero. If this is used to calculate averages, it is enough if the state sym Q is zero in those places
Eigen::Tensor<double, 1> get_cartesian_S_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 1> P(n_qubits + 1);
    for (int m = 0; m < P.dimension(0); m++) {
        P(m) = sqrt3 * (static_cast<double>(n_qubits) - 2 * m) / qubitstate_size;
    }
    return P;
}

// Returns the P function of Sx^2/S_y^2/S_z^2. They are all equal, with the only difference being which variable is spanned by the single dimension. Notice that on broadcasting only valid triples (m, n, k) should be distinct from zero. If this is used to calculate averages, it is enough if the state sym Q is zero in those places
Eigen::Tensor<double, 1> get_cartesian_S2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 1> P(n_qubits + 1);
    for (int m = 0; m < P.dimension(0); m++) {
        P(m) = ( n_qubits + xi_sum_inv*xi_sum_inv * ((static_cast<double>(n_qubits) - 2.0 * m)*(static_cast<double>(n_qubits) - 2.0 * m) - n_qubits) ) / qubitstate_size;
    }
    return P;
}

// Returns the full symmetric P function of Sx
Eigen::Tensor<double, 3> get_Sx_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = sqrt3 * (static_cast<double>(n_qubits) - 2.0 * m) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of Sy
Eigen::Tensor<double, 3> get_Sy_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = sqrt3 * (static_cast<double>(n_qubits) - 2.0 * k) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of Sz
Eigen::Tensor<double, 3> get_Sz_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = sqrt3 * (static_cast<double>(n_qubits) - 2.0 * n) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of Sx^2
Eigen::Tensor<double, 3> get_Sx2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = (n_qubits + 3 * ((static_cast<double>(n_qubits) - 2.0 * m)*(static_cast<double>(n_qubits) - 2.0 * m) - n_qubits)) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of Sy^2
Eigen::Tensor<double, 3> get_Sy2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = (n_qubits + 3 * ((static_cast<double>(n_qubits) - 2.0 * k)*(static_cast<double>(n_qubits) - 2.0 * k) - n_qubits)) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of Sz^2
Eigen::Tensor<double, 3> get_Sz2_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = (n_qubits + 3 * ((static_cast<double>(n_qubits) - 2.0 * n)*(static_cast<double>(n_qubits) - 2.0 * n) - n_qubits)) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of {Sy,Sz}/2
Eigen::Tensor<double, 3> get_aSySz_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = 3.0 * ((static_cast<double>(n_qubits) - 2.0 * k) * (static_cast<double>(n_qubits) - 2.0 * n) - (static_cast<double>(n_qubits) - 2.0 * m)) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of {Sz,Sx}/2
Eigen::Tensor<double, 3> get_aSzSx_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = 3.0 * ((static_cast<double>(n_qubits) - 2.0 * m) * (static_cast<double>(n_qubits) - 2.0 * n) - (static_cast<double>(n_qubits) - 2.0 * k)) / qubitstate_size;
    });
    return P;
}

// Returns the full symmetric P function of {Sx,Sy}/2
Eigen::Tensor<double, 3> get_aSxSy_Pfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size) {
    Eigen::Tensor<double, 3> P(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    P.setZero();
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        P(m, n, k) = 3.0 * ((static_cast<double>(n_qubits) - 2.0 * m) * (static_cast<double>(n_qubits) - 2.0 * k) - (static_cast<double>(n_qubits) - 2.0 * n)) / qubitstate_size;
    });
    return P;
}

// Calculates the average of a symmetric operator. Accepts the operator_symP as a template to allow for tensor expressions to be passed
template <typename TensorExpr>
void sym_operator_average(const TensorExpr &operator_symP, const Eigen::Tensor<double, 3> &state_symQ, double &average) {
    Eigen::Tensor<double, 0> average_result;
    // Eigen::Array<Eigen::IndexPair<int>, 3, 1> contraction_indices = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(2,2)};
    // average_result = state_symQ.contract(operator_symP, contraction_indices);
    average_result = (state_symQ * operator_symP).sum();
    average = average_result(0);
}

// Calculates the expected value of S•v, where v is given as an Eigen::Vector3d and is assummed to be normalized
void ang_operator_average(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Vector3d &v, const Eigen::Tensor<double, 3> &state_symQ, double &average) {
    average = 0;
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        average += Sv_Pfunc(n_qubits, qubitstate_size, v, m, n, k) * state_symQ(m, n, k);
    });
}

// Calculates the expected value of S•n, where n is given by its angles
void ang_operator_average(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const double &theta, const double &phi, const Eigen::Tensor<double, 3> &state_symQ, double &average) {
    ang_operator_average(n_qubits, qubitstate_size, {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)}, state_symQ, average);
}

// Calculates the averages of all quadratic operators, where the cross products are replaced by the anticommutator. Can be optimized by reducing the size of Sx/Sy/Sz to their respective variables
void cartesian_ang_operator_averages(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &state_symQ, double &Sx, double &Sy, double &Sz, double &Sx2, double &Sy2, double &Sz2, double &SySz, double &SzSx, double &SxSy) {
    static const Eigen::Tensor<double, 3> Sx_Pfunc = get_Sx_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> Sy_Pfunc = get_Sy_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> Sz_Pfunc = get_Sz_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> Sx2_Pfunc = get_Sx2_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> Sy2_Pfunc = get_Sy2_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> Sz2_Pfunc = get_Sz2_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SySz_Pfunc = get_aSySz_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SzSx_Pfunc = get_aSzSx_Pfunc(n_qubits, qubitstate_size);
    static const Eigen::Tensor<double, 3> SxSy_Pfunc = get_aSxSy_Pfunc(n_qubits, qubitstate_size);
    sym_operator_average(Sx_Pfunc, state_symQ, Sx);
    sym_operator_average(Sy_Pfunc, state_symQ, Sy);
    sym_operator_average(Sz_Pfunc, state_symQ, Sz);
    sym_operator_average(Sx2_Pfunc, state_symQ, Sx2);
    sym_operator_average(Sy2_Pfunc, state_symQ, Sy2);
    sym_operator_average(Sz2_Pfunc, state_symQ, Sz2);
    sym_operator_average(SySz_Pfunc, state_symQ, SySz);
    sym_operator_average(SzSx_Pfunc, state_symQ, SzSx);
    sym_operator_average(SxSy_Pfunc, state_symQ, SxSy);
}

// Returns the correlation matrix for a particular symmetric Q function. In order to calculate values of the Gaussian envelope, the average values of Sx, Sy, Sz are returned in their inputs as well
Eigen::Matrix3d get_correlation_matrix(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ, double &Sx, double &Sy, double &Sz) {
    double Sx2, Sy2, Sz2, SySz, SzSx, SxSy;
    cartesian_ang_operator_averages(n_qubits, qubitstate_size, symQ, Sx, Sy, Sz, Sx2, Sy2, Sz2, SySz, SzSx, SxSy);

    Eigen::Matrix3d Gamma {
        {Sx2 - Sx*Sx, SxSy - Sx*Sy, SzSx - Sz*Sx},
        {SxSy - Sx*Sy, Sy2 - Sy*Sy, SySz - Sy*Sz},
        {SzSx - Sz*Sx, SySz - Sy*Sz, Sz2 - Sz*Sz}
    };

    Eigen::Matrix3d Lambda {
        {2.0 * n_qubits, sqrt3 * Sz, sqrt3 * Sy},
        {sqrt3 * Sz, 2.0 * n_qubits, sqrt3 * Sx},
        {sqrt3 * Sy, sqrt3 * Sx, 2.0 * n_qubits}
    };
    return (Gamma + Lambda) / (6.0 * n_qubits);
}

// Returns the Gaussian envelope of the state SymQ. Needs checking, the determinant of correlation_matrix may sometimes be negative, which leads to the sqrt to be undefined.
Eigen::Tensor<double, 3> get_Gfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ) {
    Eigen::Tensor<double, 3> Gfunc(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    Gfunc.setZero();
    double Sx, Sy, Sz;
    Eigen::Matrix3d correlation_matrix = get_correlation_matrix(n_qubits, qubitstate_size, symQ, Sx, Sy, Sz);
    Eigen::Matrix3d precision_matrix = correlation_matrix.inverse();
    Eigen::Vector3d x_bar = {0.5 - Sx/(2 * sqrt3 * n_qubits), 0.5 - Sy/(2 * sqrt3 * n_qubits), 0.5 - Sz/(2 * sqrt3 * n_qubits)};
    Eigen::Vector3d x;
    double coeff = (1 << (n_qubits + 1)) / ( EIGEN_PI * n_qubits * std::sqrt( EIGEN_PI * n_qubits * correlation_matrix.determinant() ) );
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        x = {static_cast<double>(m)/n_qubits, static_cast<double>(n)/n_qubits, static_cast<double>(k)/n_qubits};
        Gfunc(m, n, k) = coeff * std::exp(- static_cast<double>(n_qubits) * (x - x_bar).transpose() * precision_matrix * (x - x_bar) );
    });
    return Gfunc;
}

// Returns both the correlation matrix for a particular symmetric Q function and all first and second symmetric moments. Could potentally rework this overload into a single functin that does this with a pointer. Not necessary for now
Eigen::Matrix3d get_correlation_matrix(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ, double &Sx, double &Sy, double &Sz, double &Sx2, double &Sy2, double &Sz2, double &SySz, double &SzSx, double &SxSy) {
    cartesian_ang_operator_averages(n_qubits, qubitstate_size, symQ, Sx, Sy, Sz, Sx2, Sy2, Sz2, SySz, SzSx, SxSy);

    Eigen::Matrix3d Gamma {
        {Sx2 - Sx*Sx, SxSy - Sx*Sy, SzSx - Sz*Sx},
        {SxSy - Sx*Sy, Sy2 - Sy*Sy, SySz - Sy*Sz},
        {SzSx - Sz*Sx, SySz - Sy*Sz, Sz2 - Sz*Sz}
    };

    Eigen::Matrix3d Lambda {
        {2.0 * n_qubits, sqrt3 * Sz, sqrt3 * Sy},
        {sqrt3 * Sz, 2.0 * n_qubits, sqrt3 * Sx},
        {sqrt3 * Sy, sqrt3 * Sx, 2.0 * n_qubits}
    };
    return (Gamma + Lambda) / (6.0 * n_qubits);
}

// Returns the Gaussian envelope of the state SymQ in Gfunc
void get_Gfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ, Eigen::Tensor<double, 3> &Gfunc) {
    Gfunc.setZero();
    double Sx, Sy, Sz;
    Eigen::Matrix3d correlation_matrix = get_correlation_matrix(n_qubits, qubitstate_size, symQ, Sx, Sy, Sz);
    Eigen::Matrix3d precision_matrix = correlation_matrix.inverse();
    Eigen::Vector3d x_bar = {0.5 - Sx/(2 * sqrt3 * n_qubits), 0.5 - Sy/(2 * sqrt3 * n_qubits), 0.5 - Sz/(2 * sqrt3 * n_qubits)};
    Eigen::Vector3d x;
    double coeff = (1 << (n_qubits + 1)) / ( EIGEN_PI * n_qubits * std::sqrt( EIGEN_PI * n_qubits * correlation_matrix.determinant() ) );
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        x = {static_cast<double>(m)/n_qubits, static_cast<double>(n)/n_qubits, static_cast<double>(k)/n_qubits};
        Gfunc(m, n, k) = coeff * std::exp(- static_cast<double>(n_qubits) * (x - x_bar).transpose() * precision_matrix * (x - x_bar) );
    });
}

// Returns the symmetrized Q function for a given state (specified by its Q function)
Eigen::Tensor<double, 3> get_symQ(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::MatrixXd &Qfunc) {
    Eigen::Tensor<double, 3> symQ(n_qubits + 1, n_qubits + 1, n_qubits + 1);
    symQ.setZero();

    unsigned int halpha = 0;
    for(unsigned int alpha = 0; alpha < qubitstate_size; alpha++) {
        halpha = std::popcount(alpha);
        for (unsigned int beta = 0; beta < qubitstate_size; beta++) {
            symQ(halpha, std::popcount(beta), std::popcount(alpha^beta)) += Qfunc(alpha, beta);
        }
    }
    return symQ;
}

// Returns the symmetrized Q function for a given state (specified by its operational basis expansion). Assumes the state has no complex phases
Eigen::Tensor<double, 3> get_symQ(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::VectorXd &state) {
    const Eigen::MatrixXd Qfunc = pure_Qfunc_from_operational(n_qubits, qubitstate_size, state);
    return get_symQ(n_qubits, qubitstate_size, Qfunc);
}

// Returns the symmetrized Q function for a given state (specified by its operational basis expansion)
Eigen::Tensor<double, 3> get_symQ(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::VectorXcd &state) {
    const Eigen::MatrixXd Qfunc = pure_Qfunc_from_operational(n_qubits, qubitstate_size, state);
    return get_symQ(n_qubits, qubitstate_size, Qfunc);
}

// Saves the symQfunc in filename using a csv format
void save_symQfunc(const Eigen::Tensor<double,3> &symQfunc, const std::string &filename) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::ofstream output_file(cwd.string()+"/data/symQfuncs/"+filename,std::ofstream::out|std::ofstream::ate|std::ofstream::trunc);
    if (output_file.is_open()) {
        Eigen::TensorIOFormat csv_format = Eigen::TensorIOFormat(/*separator=*/{",\n", ""}, /*prefix=*/{"", ""}, /*suffix=*/{"", ""}, /*precision=*/Eigen::FullPrecision, /*flags=*/0, /*tenPrefix=*/"", /*tenSuffix=*/"");
        output_file << symQfunc.format(csv_format);
    } else {
        std::cout << "Could not save symQfunc" << std::endl;
    }
}

// Returns the Kravchuck expansion and the Gaussian envelope of the state SymQ. Needs checking, the determinant of correlation_matrix may sometimes be negative, which leads to the sqrt to be undefined.
void get_Kravchuk_expansion_Gfunc(const unsigned int &n_qubits, const unsigned int &qubitstate_size, const Eigen::Tensor<double, 3> &symQ, Eigen::Tensor<double, 3> &Gfunc, Eigen::Tensor<double, 3> &Kravchuk_exp) {
    Gfunc.setZero();
    double Sx, Sy, Sz, Sx2, Sy2, Sz2, SySz, SzSx, SxSy;
    Eigen::Matrix3d correlation_matrix = get_correlation_matrix(n_qubits, qubitstate_size, symQ, Sx, Sy, Sz, Sx2, Sy2, Sz2, SySz, SzSx, SxSy);
    Eigen::Matrix3d precision_matrix = correlation_matrix.inverse();
    Eigen::Vector3d x_bar = {0.5 - Sx/(2 * sqrt3 * n_qubits), 0.5 - Sy/(2 * sqrt3 * n_qubits), 0.5 - Sz/(2 * sqrt3 * n_qubits)};
    Eigen::Vector3d x;
    double coeff = (1 << (n_qubits + 1)) / ( EIGEN_PI * n_qubits * std::sqrt( EIGEN_PI * n_qubits * correlation_matrix.determinant() ) );
    sym_space_loop(n_qubits, [&](int &m, int &n, int &k) {
        x = {static_cast<double>(m)/n_qubits, static_cast<double>(n)/n_qubits, static_cast<double>(k)/n_qubits};
        Gfunc(m, n, k) = coeff * std::exp(- static_cast<double>(n_qubits) * (x - x_bar).transpose() * precision_matrix * (x - x_bar) );
    });
    Eigen::Tensor<double, 3> Nm, Nn, Nk, K1m, K1n, K1k, m, n, k;
    const Eigen::Tensor<double, 1> binomial = binom(n_qubits);
    Eigen::VectorXd seq = Eigen::VectorXd::EqualSpaced(n_qubits+1, 0, 1);
    Eigen::TensorMap<Eigen::Tensor<double, 1>> seq_tensor(seq.data(), n_qubits+1);
    Eigen::array<Eigen::Index, 3> new_shape = {n_qubits+1, 1, 1};
    Eigen::array<Eigen::Index, 3> bcast = {1, n_qubits+1, n_qubits+1};
    Nm = binomial.reshape(new_shape).broadcast(bcast);
    m = seq_tensor.reshape(new_shape).broadcast(bcast);
    new_shape = {1, n_qubits+1, 1};
    bcast = {n_qubits+1, 1, n_qubits+1};
    Nn = ( 1.0 / static_cast<double>(1 << n_qubits) ) * binomial.reshape(new_shape).broadcast(bcast); // N choose n and N choose k include the 2^{-N} in front of them to reduce possible overflow
    n = seq_tensor.reshape(new_shape).broadcast(bcast);
    new_shape = {1, 1, n_qubits+1};
    bcast = {n_qubits+1, n_qubits+1, 1};
    Nk = ( 1.0 / static_cast<double>(1 << n_qubits) ) * binomial.reshape(new_shape).broadcast(bcast);
    k = seq_tensor.reshape(new_shape).broadcast(bcast);
    K1m = - (2.0/n_qubits) * m + static_cast<double>(1);
    K1n = - (2.0/n_qubits) * n + static_cast<double>(1);
    K1k = - (2.0/n_qubits) * k + static_cast<double>(1);
    Kravchuk_exp = Nm * Nn * Nk * ( (Sx/sqrt3) * K1m + (Sz/sqrt3) * K1n * n) + (Sy/sqrt3) * K1k + (1.0 / (3 * n_qubits *(n_qubits - 1))) * ( (Sx2 - n_qubits)*(2*m*m - 2*n_qubits*m + static_cast<double>(n_qubits*(n_qubits-1)/2)) + (Sz2 - n_qubits)*(2*n*n - 2*n_qubits*n + static_cast<double>(n_qubits*(n_qubits-1)/2)) + (Sy2 - n_qubits)*(2*k*k - 2*n_qubits*k + static_cast<double>(n_qubits*(n_qubits-1)/2)) ) + (1.0/6)*( (SySz + 2*sqrt3*Sx)*K1n*K1k + (SxSy + 2*sqrt3*Sz)*K1m*K1k + (SzSx + 2*sqrt3*Sy)*K1m*K1n + static_cast<double>(1));
    Kravchuk_exp *= sym_space_mask(n_qubits);
}