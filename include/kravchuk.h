#ifndef KRAVCHUK_H
#define KRAVCHUK_H

#include<vector>
#include<memory>
#include<unsupported/Eigen/CXX11/Tensor>

class polynomial{
public:
    // Builds a polynomial based on its coefficients
    polynomial(unsigned int const &rank, std::unique_ptr<double[]> const &in_coeffs);
    // Copy constructor
    polynomial(polynomial const &other);
    //  Builds the next Kravchuk polynomial from Kravchuk recurrence relations
    polynomial(polynomial const &first, polynomial const &second, int const &N);
    // Prints the polynomial to console in readable form
    void print() const;
    // Returns n-th coefficient of this
    double operator[](unsigned int const &n) const;
    // The rank of this
    unsigned int rank() const;
    // Returns this evaluated at x
    double operator()(int const &x) const;
private:
    unsigned int const n_rank;
    std::unique_ptr<double[]> coeffs;
};

//  Builds the next Kravchuk polynomial from their recurrence relations
polynomial get_next_Kravchuk(polynomial const &first, polynomial const &second, unsigned int const &N);

// Returns all Kravchuk polynomials from rank 0 to rank max_rank, with N fixed
std::vector<polynomial> get_Kravchuk_pols(unsigned int const &max_rank, unsigned int const &N);

//  Builds the next Kravchuk polynomial from their recurrence relations
polynomial get_next_Kravchuk(polynomial const &first, polynomial const &second, unsigned int const &N);

// Returns all Kravchuk polynomials from rank 0 to rank max_rank, with N fixed
std::vector<polynomial> get_Kravchuk_pols(unsigned int const &max_rank, unsigned int const &N);

// Polynomial over 3 variables
class polynomial3{
public:
    // Build a polynomial of fixed rank, with all coefficients set to zero
    polynomial3(unsigned int const &rank1, unsigned int const &rank2, unsigned int const &rank3);
    // Builds a polynomial from its coefficients. Invalidates in_coeffs to avoid copying when possible
    polynomial3(unsigned int const &rank1, unsigned int const &rank2, unsigned int const &rank3, std::unique_ptr<double[]> &in_coeffs);
    // Copy constructor
    polynomial3(polynomial3 const &other);
    // Move constructor
    polynomial3(polynomial3 &&other);
    // Constructs a polynomial over three variables from three polynomials, each over one variable
    polynomial3(polynomial const &p1, polynomial const &p2, polynomial const &p3);
    // Prints this for debugging
    void const print() const;
    // Returns this evaluated at (m,n,k)
    double eval(int const &m, int const &n, int const &k) const;
    // Calculates this polynomial multiplied by Binom(N, m) * Binom(N, n) * Binom(N, k)
    double binom_eval(unsigned int const &N, int const &m, int const &n, int const &k) const;
    // Multiplies all coefficients of this by scalar
    polynomial3& mult(double const &scalar) &;
    // Multiplies all coefficients of this by scalar
    polynomial3&& mult(double const &scalar) &&;
    // Sums all coefficients of other, multiplied by scalar, to this
    polynomial3& sum_mult(polynomial3 const& other, double const &scalar) &;
    // Sums all coefficients of other, multiplied by scalar, to this
    polynomial3&& sum_mult(polynomial3 const& other, double const &scalar) &&;
    // Sets all coefficients of this to zero
    polynomial3& set_zero() &;
    // Sets all coefficients of this to zero
    polynomial3&& set_zero() &&;
    // Returns this polynomial as a tensor, evaluated over all symmetric space
    Eigen::Tensor<double, 3> as_tensor(unsigned int const &n_qubits) const;
    // Returns this polynomial as a tensor, evaluated over all symmetric space with leading binomials Binom(N, m) * Binom(N, n) * Binom(N, k)
    Eigen::Tensor<double, 3> as_binom_tensor(unsigned int const &n_qubits) const;
    // Assignment operator, moves other.coeffs to this
    polynomial3& operator=(polynomial3 &&other);
    // Returns a reference to the (m,n,k)-th power coefficient of this
    double& operator()(int const &m, int const &n, int const &k);
    // Returns a const reference to the (m,n,k)-th power coefficient of this
    double const& operator()(int const &m, int const &n, int const &k) const;
    // Adds other to this as polynomials
    void operator+=(polynomial3 const &other);
    // Returns a new polynomial, with coefficients equal to the sum of coefficients of both this and other
    polynomial3 operator+(polynomial3 const &other);
private:
    unsigned int n_rank1, n_rank2, n_rank3;
    std::unique_ptr<double[]> coeffs;
};

// 
class kravchuk_exp{
public:
    // Build a kravchuk expansion for a symmetric function of n_qubits, with all coefficients set to zero
    kravchuk_exp(unsigned int const &n_qubits);
    // Builds a kravchuk expansion for a symmetric function of n_qubits. Copies in_coeffs as coefficients
    kravchuk_exp(unsigned int const &n_qubits, Eigen::Tensor<double, 3> &in_coeffs);
    // Builds a kravchuk expansion for a symmetric function of n_qubits, from an rvalue tensor as coefficients
    kravchuk_exp(unsigned int const &n_qubits, Eigen::Tensor<double, 3> &&in_coeffs);
    // Copy constructor
    kravchuk_exp(kravchuk_exp const &other);
    // Move constructor
    kravchuk_exp(kravchuk_exp &&other);
    // Prints this for debugging
    void const print() const;
    // Returns this evaluated at (m,n,k)
    double eval(int const &m, int const &n, int const &k) const;
    // Calculates this evaluated at (m,n,k), multiplied by Binom(N, m) * Binom(N, n) * Binom(N, k)
    double binom_eval(int const &m, int const &n, int const &k) const;
    // Multiplies all coefficients of this by scalar
    kravchuk_exp& mult(double const &scalar) &;
    // Multiplies all coefficients of this by scalar
    kravchuk_exp&& mult(double const &scalar) &&;
    // Sums all coefficients of other, multiplied by scalar, to this. Only does so if n_qubits for this and other is the same, otherwise it does nothing
    kravchuk_exp& sum_mult(kravchuk_exp const& other, double const &scalar) &;
    // Sums all coefficients of other, multiplied by scalar, to this. Only does so if n_qubits for this and other is the same, otherwise it does nothing
    kravchuk_exp&& sum_mult(kravchuk_exp const& other, double const &scalar) &&;
    // Sets all coefficients of this to zero
    kravchuk_exp& set_zero() &;
    // Sets all coefficients of this to zero
    kravchuk_exp&& set_zero() &&;
    // Copies in_coeffs to this.coeffs
    kravchuk_exp& set_coeffs(Eigen::Tensor<double, 3> &in_coeffs) &;
    // Moves in_coeffs to this.coeffs
    kravchuk_exp& set_coeffs(Eigen::Tensor<double, 3> &&in_coeffs) &;
    // Copies in_coeffs to this.coeffs
    kravchuk_exp&& set_coeffs(Eigen::Tensor<double, 3> &in_coeffs) &&;
    // Moves in_coeffs to this.coeffs
    kravchuk_exp&& set_coeffs(Eigen::Tensor<double, 3> &&in_coeffs) &&;
    // Returns this expansion as a tensor, evaluated over all symmetric space
    Eigen::Tensor<double, 3> as_tensor() const;
    // Returns this expansion as a tensor, evaluated over all symmetric space with leading binomials Binom(N, m) * Binom(N, n) * Binom(N, k)
    Eigen::Tensor<double, 3> as_binom_tensor() const;
    // Move assignment operator
    kravchuk_exp& operator=(kravchuk_exp &&other);
    // Returns a reference to the (m,n,k)-th coefficient of this
    double& operator()(int const &m, int const &n, int const &k);
    // Returns a const reference to the (m,n,k)-th coefficient of this
    double const& operator()(int const &m, int const &n, int const &k) const;
    // Adds other to this as expansions
    void operator+=(kravchuk_exp const &other);
    // Returns a new kravchuk_exp, with coefficients equal to the sum of coefficients of both this and other. Only does so if n_qubits for this and other is the same, otherwise it returns an empy kravchuk_exp with n_qubits = this.n_qubits
    kravchuk_exp operator+(kravchuk_exp const &other);
private:
    Eigen::Tensor<double, 3> coeffs;
    unsigned int n_qubits;
    std::vector<polynomial> kravchuks;
    std::vector<double> binoms;
};

#endif