#include "GF2N_matrix.h"
#include<iostream>
#include<bitset>
#include "GF2N.h"

// Builds a GF2N matrix A, with size entries where the k-th bit (from right) of the j-th entry corresponds to A_jk. Sets all rows to def
GF2N_matrix::GF2N_matrix(unsigned int const &size, unsigned int const &def) : size(size), rows(std::make_unique<unsigned int[]>(size)) {
    for (int j = 0; j < size; j++) {
        rows[j] = def;
    }
}

// Copies matrix as a GF2N matrix A, with size entries where the k-th bit (from right) of the j-th entry corresponds to A_jk
GF2N_matrix::GF2N_matrix(unsigned int const &size, unsigned int const *matrix) : size(size), rows(std::make_unique<unsigned int[]>(size)) {
    for (int j = 0; j < size; j++) {
        rows[j] = matrix[j];
    }
}

// Copies matrix as a GF2N matrix A, with size entries where the k-th bit (from right) of the j-th entry corresponds to A_jk. Invalidates matrix to avoid copying when possible
GF2N_matrix::GF2N_matrix(unsigned int const &size, std::unique_ptr<unsigned int[]> &matrix) : size(size) {
    rows = std::move(matrix);
}

// Copy constructor
GF2N_matrix::GF2N_matrix(GF2N_matrix const &other) : size(other.size), rows(std::make_unique<unsigned int[]>(size)) {
    for (int j = 0; j < size; j++) {
        rows[j] = other.rows[j];
    }
}

// Move constructor
GF2N_matrix::GF2N_matrix(GF2N_matrix &&other) : size(size) {
    rows = std::move(other.rows);
}

// Prints this for debugging
void const GF2N_matrix::print() const {
    for (int j = 0; j < size; j++) {
        std::cout << (*this)(j) << std::endl;
    }
    std::cout << "_____________" << std::endl;
}

// Returns vec*this, where vec is treated as a GF2N covector
unsigned int GF2N_matrix::lmult(unsigned int const &vec) const {
    unsigned int res = 0;
    for (int j = 0; j < size; j++) {
        res ^= get_bit(vec, j) * rows[j];
    }
    return res;
}

// Returns this*other, as a GF2N matrix product. If other.size != size, it returns a 0 matrix
GF2N_matrix GF2N_matrix::mult(GF2N_matrix const &other) const {
    GF2N_matrix mult(size, 0u);
    if (size == other.size) {
        unsigned int left_row = 0;
        for (int j = 0; j < size; j++) {
            left_row = rows[j];
            for (int k = 0; k < size; k++) {
                mult[j] ^= get_bit(left_row, k) * other[k];
            }
        }
    }
    return mult;
}

// Sets all rows of this to zero
GF2N_matrix& GF2N_matrix::set_zero() & {
    for (int j = 0; j < size; j++) {
        rows[j] = 0;
    }
    return *this;
}

// Sets all rows of this to zero
GF2N_matrix&& GF2N_matrix::set_zero() && {
    for (int j = 0; j < size; j++) {
        rows[j] = 0;
    }
    return std::move(*this);
}

// Copies in_coeffs to this.coeffs. Invalidates in_rows to avoid copying
GF2N_matrix& GF2N_matrix::set_coeffs(std::unique_ptr<unsigned int[]> &in_rows) & {
    rows = std::move(in_rows);
    return *this;
}

// Copies in_coeffs to this.coeffs. Invalidates in_rows to avoid copying
GF2N_matrix&& GF2N_matrix::set_coeffs(std::unique_ptr<unsigned int[]> &in_rows) && {
    rows = std::move(in_rows);
    return std::move(*this);
}

// Assignment operator, moves other.coeffs to this. If other.size != size, it does nothing
GF2N_matrix& GF2N_matrix::operator=(GF2N_matrix &&other) {
    if (other.size == size) {
        rows = std::move(other.rows);
    }
    return *this;
}

// Returns a reference to the j-th row of this
unsigned int& GF2N_matrix::operator[](int const &j) {
    return rows[j];
}

// Returns a const reference to the j-th row of this
unsigned int const& GF2N_matrix::operator[](int const &j) const {
    return rows[j];
}

// Returns the Aj,k-th bit of this as an unsigned int
unsigned int GF2N_matrix::operator[](int const &j, int const &k) const {
    return get_bit(rows[j], k);
}

// Returns the j-th row of this as an Eigen::VectorXi
Eigen::VectorXi GF2N_matrix::operator()(int const &j) const {
    Eigen::VectorXi row(size);
    for (int k = 0; k < size; k++) {
        row[k] = get_bit(rows[j], k);
    }
    return row;
}

// Returns this XOR other, as a new GF2N_matrix. If other.size != size, it returns a 0 matrix
GF2N_matrix GF2N_matrix::operator^(GF2N_matrix const &other) {
    GF2N_matrix res(size, 0u);
    if (size == other.size) {
        for (int j = 0; j < size; j++) {
            res[j] = rows[j] ^ other[j];
        }
    }
    return res;
}

// Applies XOR other to this in place. If other.size != size, it does nothing
GF2N_matrix& GF2N_matrix::operator^=(GF2N_matrix const &other) {
    if (size == other.size) {
        for (int j = 0; j < size; j++) {
            rows[j] ^= other[j];
        }
    }
    return *this;
}

// Returns this*other, as a GF2N matrix product. If other.size != size, it returns a 0 matrix
GF2N_matrix GF2N_matrix::operator*(GF2N_matrix const &other) {
    GF2N_matrix mult(size, 0u);
    if (size == other.size) {
        unsigned int left_row = 0;
        for (int j = 0; j < size; j++) {
            left_row = rows[j];
            for (int k = 0; k < size; k++) {
                mult[j] ^= get_bit(left_row, k) * other[k];
            }
        }
    }
    return mult;
}

// Multiplies this*other in place. If other.size != size, it does nothing
GF2N_matrix& GF2N_matrix::operator*=(GF2N_matrix const &other) {
    if (size == other.size) {
        unsigned int left_row = 0;
        for (int j = 0; j < size; j++) {
            left_row = rows[j];
            rows[j] = 0;
            for (int k = 0; k < size; k++) {
                rows[j] ^= get_bit(left_row, k) * other[k];
            }
        }
    }
    return *this;
}