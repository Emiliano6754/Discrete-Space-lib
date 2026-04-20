#include "GF2N_matrix.h"
#include<iostream>
#include<bitset>
#include<bit>
#include "GF2N.h"

// Builds a GF2N matrix A, with size entries where the k-th bit (from right) of the j-th entry corresponds to A_jk. Sets all rows to the rightmost n(=columns)-bits of def
GF2N_matrix::GF2N_matrix(unsigned int const &rows, unsigned int const &columns, unsigned int const &def) : rows_n(rows), columns_n(columns), matrix_rows(std::make_unique<unsigned int[]>(rows_n)) {
    unsigned int mask = (1 << columns_n) - 1;
    for (int j = 0; j < rows_n; j++) {
        matrix_rows[j] = def & mask;
    }
}

// Copies matrix as a GF2N matrix A, with size entries where the k-th bit (from right) of the j-th entry corresponds to A_jk
GF2N_matrix::GF2N_matrix(unsigned int const &rows, unsigned int const &columns, unsigned int const *matrix) : rows_n(rows), columns_n(columns), matrix_rows(std::make_unique<unsigned int[]>(rows_n)) {
    unsigned int mask = (1 << columns_n) - 1;
    for (int j = 0; j < rows_n; j++) {
        matrix_rows[j] = matrix[j] & mask;
    }
}

// Copies matrix as a GF2N matrix A, with size entries where the k-th bit (from right) of the j-th entry corresponds to A_jk. Invalidates matrix to avoid copying when possible
GF2N_matrix::GF2N_matrix(unsigned int const &rows, unsigned int const &columns, std::unique_ptr<unsigned int[]> &matrix) : rows_n(rows), columns_n(columns) {
    matrix_rows = std::move(matrix);
}

// Copy constructor
GF2N_matrix::GF2N_matrix(GF2N_matrix const &other) : rows_n(other.rows_n), columns_n(other.columns_n), matrix_rows(std::make_unique<unsigned int[]>(rows_n)) {
    for (int j = 0; j < rows_n; j++) {
        matrix_rows[j] = other.matrix_rows[j];
    }
}

// Move constructor
GF2N_matrix::GF2N_matrix(GF2N_matrix &&other) : rows_n(rows_n), columns_n(columns_n) {
    matrix_rows = std::move(other.matrix_rows);
}

// Prints this for debugging
void const GF2N_matrix::print() const {
    for (int j = 0; j < rows_n; j++) {
        std::cout << (*this)(j) << std::endl;
    }
    std::cout << "_____________" << std::endl;
}

// Returns vec*this, where vec is treated as a GF2N covector
unsigned int GF2N_matrix::lmult(unsigned int const &vec) const {
    unsigned int res = 0;
    for (int j = 0; j < rows_n; j++) {
        res ^= get_bit(vec, j) * matrix_rows[j];
    }
    return res;
}

// Returns this*other, as a GF2N matrix product. If other.rows != columns, it returns a 0 matrix
GF2N_matrix GF2N_matrix::mult(GF2N_matrix const &other) const {
    GF2N_matrix mult(rows_n, other.columns_n, 0u);
    if (columns_n == other.rows_n) {
        unsigned int left_row = 0;
        for (int j = 0; j < rows_n; j++) {
            left_row = matrix_rows[j];
            for (int k = 0; k < columns_n; k++) {
                mult[j] ^= get_bit(left_row, k) * other[k];
            }
        }
    }
    return mult;
}

// Sets all rows of this to zero
GF2N_matrix& GF2N_matrix::set_zero() & {
    for (int j = 0; j < rows_n; j++) {
        matrix_rows[j] = 0;
    }
    return *this;
}

// Sets all rows of this to zero
GF2N_matrix&& GF2N_matrix::set_zero() && {
    for (int j = 0; j < rows_n; j++) {
        matrix_rows[j] = 0;
    }
    return std::move(*this);
}

// Copies in_coeffs to this.coeffs. Invalidates in_rows to avoid copying
GF2N_matrix& GF2N_matrix::set_coeffs(std::unique_ptr<unsigned int[]> &in_rows) & {
    matrix_rows = std::move(in_rows);
    return *this;
}

// Copies in_coeffs to this.coeffs. Invalidates in_rows to avoid copying
GF2N_matrix&& GF2N_matrix::set_coeffs(std::unique_ptr<unsigned int[]> &in_rows) && {
    matrix_rows = std::move(in_rows);
    return std::move(*this);
}

// Returns the inverse of this matrix, by row-reduction. If this is not invertible, it is not a true inverse. This is not well behaved for non-square matrices, because of << rows_n
GF2N_matrix GF2N_matrix::inverse() const {
    GF2N_matrix inverse(rows_n, columns_n, 0u);
    unsigned long long* augmented_matrix = static_cast<unsigned long long*>( _malloca(rows_n * sizeof(unsigned long long)) );
    for (unsigned int j = 0; j < rows_n; j++) {
        augmented_matrix[j] = (static_cast<unsigned long long>(matrix_rows[j]) << rows_n) | (1 << j);
    }
    unsigned long long pivot = 1 << (2*rows_n - 1);
    for (unsigned int j = 0; j < rows_n; j++) {
        // Find pivot elements and order them accordingly 
        for (unsigned int k = j; k < rows_n; k++) {
            if (augmented_matrix[k] & pivot) {
                std::swap(augmented_matrix[j], augmented_matrix[k]);
                break;
            }
        }
        // Eliminate the remaining ones in that pivot
        for (unsigned int k = 0; k < rows_n; k++) {
            if (augmented_matrix[k] & pivot && k != j) {
                augmented_matrix[k] ^= augmented_matrix[j];
            }
        }
        pivot >>= 1;
    }
    unsigned long long mask = (1 << rows_n) - 1;
    for (unsigned int j = 0; j < rows_n; j++) {
        inverse[j] = static_cast<unsigned int>(augmented_matrix[j] & mask);
    }
    return inverse;
}

// Returns the row-echelon form of this matrix
GF2N_matrix GF2N_matrix::row_echelon() const {
    GF2N_matrix row_echelon(*this);
    unsigned int pivot = 1 << (columns_n - 1);
    for (unsigned int j = 0; j < columns_n; j++) {
        // Find pivot elements and order them accordingly 
        for (unsigned int k = j; k < rows_n; k++) {
            if (row_echelon[k] & pivot) {
                std::swap(row_echelon[j], row_echelon[k]);
                break;
            }
        }
        // Eliminate the remaining ones in that pivot
        for (unsigned int k = 0; k < rows_n; k++) {
            if (row_echelon[k] & pivot && k != j) {
                row_echelon[k] ^= row_echelon[j];
            }
        }
        pivot >>= 1;
    }
    return row_echelon;
}

void GF2N_matrix::row_reduce() {
    unsigned int pivot = 1 << (columns_n - 1);
    for (unsigned int j = 0; j < columns_n; j++) {
        // Find pivot elements and order them accordingly 
        for (unsigned int k = j; k < rows_n; k++) {
            if (matrix_rows[k] & pivot) {
                std::swap(matrix_rows[j], matrix_rows[k]);
                break;
            }
        }
        // Eliminate the remaining ones in that pivot
        for (unsigned int k = 0; k < rows_n; k++) {
            if (matrix_rows[k] & pivot && k != j) {
                matrix_rows[k] ^= matrix_rows[j];
            }
        }
        pivot >>= 1;
    }
}

unsigned int GF2N_matrix::rank() const {
    GF2N_matrix row_echelon = this->row_echelon();
    unsigned int rank = 0;
    for (int j = 0; j < rows_n; j++) {
        rank += (std::popcount(row_echelon[j]) > 0);
    }
    return rank;
}

// Returns (rows_n, columns_n) as an std::pair<unsigned int>
std::pair<unsigned int, unsigned int> GF2N_matrix::size() const {
    return std::pair<unsigned int, unsigned int>(rows_n, columns_n);
}

// Assignment operator, moves other.coeffs to this. If other.size != size, it does nothing
GF2N_matrix& GF2N_matrix::operator=(GF2N_matrix &&other) {
    if (other.rows_n == rows_n) {
        matrix_rows = std::move(other.matrix_rows);
    }
    return *this;
}

// Returns a reference to the j-th row of this
unsigned int& GF2N_matrix::operator[](int const &j) {
    return matrix_rows[j];
}

// Returns a const reference to the j-th row of this
unsigned int const& GF2N_matrix::operator[](int const &j) const {
    return matrix_rows[j];
}

// Returns the Aj,k-th bit of this as an unsigned int
unsigned int GF2N_matrix::operator()(int const &j, int const &k) const {
    return get_bit(matrix_rows[j], k);
}

// Returns the j-th row of this as an Eigen::VectorXi
Eigen::VectorXi GF2N_matrix::operator()(int const &j) const {
    Eigen::VectorXi row(columns_n);
    for (int k = 0; k < columns_n; k++) {
        row[k] = get_bit(matrix_rows[j], k);
    }
    return row;
}

// Returns this XOR other, as a new GF2N_matrix. If other.size != size, it returns a 0 matrix
GF2N_matrix GF2N_matrix::operator^(GF2N_matrix const &other) {
    GF2N_matrix res(rows_n, columns_n, 0u);
    if (this->size() == other.size()) {
        for (int j = 0; j < rows_n; j++) {
            res[j] = matrix_rows[j] ^ other[j];
        }
    }
    return res;
}

// Applies XOR other to this in place. If other.size != size, it does nothing
GF2N_matrix& GF2N_matrix::operator^=(GF2N_matrix const &other) {
    if (this->size() == other.size()) {
        for (int j = 0; j < rows_n; j++) {
            matrix_rows[j] ^= other[j];
        }
    }
    return *this;
}

// Returns this*other, as a GF2N matrix product. If other.rows != columns, it returns a 0 matrix
GF2N_matrix GF2N_matrix::operator*(GF2N_matrix const &other) {
    GF2N_matrix mult(rows_n, other.columns_n, 0u);
    if (columns_n == other.rows_n) {
        unsigned int left_row = 0;
        for (int j = 0; j < rows_n; j++) {
            left_row = matrix_rows[j];
            for (int k = 0; k < columns_n; k++) {
                mult[j] ^= get_bit(left_row, k) * other[k];
            }
        }
    }
    return mult;
}

// Multiplies this*other in place. If other is not square (other.rows_n != other.size_n), it does nothing
GF2N_matrix& GF2N_matrix::operator*=(GF2N_matrix const &other) {
    if (columns_n == other.rows_n && other.rows_n == other.columns_n) {
        unsigned int left_row = 0;
        for (int j = 0; j < rows_n; j++) {
            left_row = matrix_rows[j];
            matrix_rows[j] = 0;
            for (int k = 0; k < columns_n; k++) {
                matrix_rows[j] ^= get_bit(left_row, k) * other[k];
            }
        }
    }
    return *this;
}