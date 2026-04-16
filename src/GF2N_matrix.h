#ifndef GF2N_MATRIX
#define GF2N_MATRIX
#include<memory>
#include<Eigen/Dense>

class GF2N_matrix {
public:
    // Builds a GF2N matrix A, with size entries where the k-th bit (from right) of the j-th entry corresponds to A_jk. Sets all rows to def. For def = 0, must use def = 0u, so that no ambiguity arises with the uint* version 
    GF2N_matrix(unsigned int const &size, unsigned int const &def);
    // Copies matrix as a GF2N matrix A, with size entries where the k-th bit (from right) of the j-th entry corresponds to A_jk
    GF2N_matrix(unsigned int const &size, unsigned int const *matrix);
    // Copies matrix as a GF2N matrix A, with size entries where the k-th bit (from right) of the j-th entry corresponds to A_jk. Invalidates matrix to avoid copying when possible
    GF2N_matrix(unsigned int const &size, std::unique_ptr<unsigned int[]> &matrix);
    // Copy constructor
    GF2N_matrix(GF2N_matrix const &other);
    // Move constructor
    GF2N_matrix(GF2N_matrix &&other);
    // Prints this for debugging
    void const print() const;
    // Returns vec*this, where vec is treated as a GF2N vector 
    unsigned int lmult(unsigned int const &vec) const;
    // Returns this*other, as a GF2N matrix product. If other.size != size, it returns a 0 matrix
    GF2N_matrix mult(GF2N_matrix const &other) const;
    // Sets all rows of this to zero
    GF2N_matrix& set_zero() &;
    // Sets all rows of this to zero
    GF2N_matrix&& set_zero() &&;
    // Copies in_coeffs to this.coeffs. Invalidates in_rows to avoid copying
    GF2N_matrix& set_coeffs(std::unique_ptr<unsigned int[]> &in_rows) &;
    // Copies in_coeffs to this.coeffs. Invalidates in_rows to avoid copying
    GF2N_matrix&& set_coeffs(std::unique_ptr<unsigned int[]> &in_rows) &&;
    // Assignment operator, moves other.coeffs to this. If other.size != size, it does nothing
    GF2N_matrix& operator=(GF2N_matrix &&other);
    // Returns a reference to the j-th row of this
    unsigned int& operator[](int const &j);
    // Returns a const reference to the j-th row of this
    unsigned int const& operator[](int const &j) const;
    // Returns the j,k-th bit of this as an unsigned int
    unsigned int operator[](int const &j, int const &k) const;
    // Returns the j-th row of this as an Eigen::VectorXi
    Eigen::VectorXi operator()(int const &j) const;
    // Returns this XOR other, as a new GF2N_matrix
    GF2N_matrix operator^(GF2N_matrix const &other);
    // Applies XOR other to this in place
    GF2N_matrix& operator^=(GF2N_matrix const &other);
    // Returns this*other, as a GF2N matrix product
    GF2N_matrix operator*(GF2N_matrix const &other);
    // Multiplies this*other in place
    GF2N_matrix& operator*=(GF2N_matrix const &other);
private:
    std::unique_ptr<unsigned int[]> rows;
    unsigned int const size;
};

#endif