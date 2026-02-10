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
    // Prints the polynomial in readable form
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

// Polynomial over 3 variables
class polynomial3{
public:
    // Build a polynomial of fixed rank, with all coefficients set to zero
    polynomial3(unsigned int const &rank1, unsigned int const &rank2, unsigned int const &rank3);
    // Builds a polynomial from its coefficients
    polynomial3(unsigned int const &rank1, unsigned int const &rank2, unsigned int const &rank3, std::unique_ptr<double[]> const &in_coeffs);
    // Copy constructor
    polynomial3(polynomial3 const &other);
    // Constructs a polynomial over three variables from three polynomials, each over one variable
    polynomial3(polynomial const &p1, polynomial const &p2, polynomial const &p3);
    // Returns this evaluated at (m,n,k)
    double const eval(int const &m, int const &n, int const &k) const;
    // Calculates this polynomial multiplied by Binom(N, m) * Binom(N, n) * Binom(N, k)
    double binom_eval(unsigned int const &N, int const &m, int const &n, int const &k) const;
    // Multiplies all coefficients of this by scalar
    polynomial3& mult(double const &scalar) &;
    // Multiplies all coefficients of this by scalar
    polynomial3&& mult(double const &scalar) &&;
    // Sets all coefficients of this to zero
    polynomial3& set_zero() &;
    // Sets all coefficients of this to zero
    polynomial3&& set_zero() &&;
    // Returns this polynomial as a tensor, evaluated over all symmetric space
    Eigen::Tensor<double, 3> as_tensor(unsigned int &n_qubits) const;
    // Returns this polynomial as a tensor, evaluated over all symmetric space with leading binomials Binom(N, m) * Binom(N, n) * Binom(N, k)
    Eigen::Tensor<double, 3> as_binom_tensor(unsigned int &n_qubits) const;
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

//  Builds the next Kravchuk polynomial from their recurrence relations
polynomial get_next_Kravchuk(polynomial const &first, polynomial const &second, unsigned int const &N);

// Returns all Kravchuk polynomials from rank 0 to rank max_rank, with N fixed
std::vector<polynomial> get_Kravchuk_pols(unsigned int const &max_rank, unsigned int const &N);

//  Builds the next Kravchuk polynomial from their recurrence relations
polynomial get_next_Kravchuk(polynomial const &first, polynomial const &second, unsigned int const &N);

// Returns all Kravchuk polynomials from rank 0 to rank max_rank, with N fixed
std::vector<polynomial> get_Kravchuk_pols(unsigned int const &max_rank, unsigned int const &N);

#endif