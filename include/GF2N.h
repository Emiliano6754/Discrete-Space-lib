#ifndef GF2NH
#define GF2NH
#include<x86intrin.h>
#include<string>

// Returns the number on the (N-2)th row of polynomial filename. Can be used to parse the full, reduced and reducing polynomials. Can read polynomials of up to 64 qubits
__m128i read_polynomial(const std::string &polynomial_filename, const unsigned int &N);

// This should work as long as there are less than 32 qubits. For N > 32, the getters (_mm_cvtsi128_si64) must take into account that some part of the numbers is also stored in the other 64 bit half. The setters in first and second are also wrong for this
unsigned int GF2N_pol_mult(const unsigned int &a, const unsigned int &b, const unsigned int &N);

// Returns in base the coefficients in the generator basis of a self-dual basis. Can only hold 32 qubits bases and the N = 1 qubits is trivialized, so that this returns an error in that case. Assumes base points to N*sizeof(unsigned int) allocated space in memory
void read_basis_from_generator(const std::string &basis_filename, unsigned int* basis, const unsigned int &N);

// Changes element to another basis, given the expansion of the current basis in terms of the new basis. This expansion is assumed to be contained in basis, with N elements.
void change_basis(const unsigned int &element, const unsigned int* basis, unsigned int &transformed_element, const unsigned int &N);

// Returns element expanded in another basis, given the expansion of the current basis in terms of the new basis. This expansion is assumed to be contained in basis, with N elements.
unsigned int change_basis(const unsigned int &element, const unsigned int* basis, const unsigned int &N);

// Calculates the inverse of matrix, where the binary decomposition of each element is taken as a row. Assumes both matrix and inverse_matrix hold space for N elements. Only works for up to 32 qubits. This can be seen in that it takes only unsigned ints, so that no error can happen. Augmented matrix may be changed to two separate matrices, for which more qubits can be added
void GF2N_invert_matrix(const unsigned int* matrix, unsigned int* inverse_matrix, const unsigned int &N);

// Reads the self dual basis expanded in the polynomial basis from database
void initialize_self_dual_basis(unsigned int* self_dual_basis, unsigned int* generator_basis, const unsigned int &N);

// Checks if a set of basis vector indices (j_1, j_2, ..., j_r) has all indices set to n_qubits
bool check_j_vector_max(const unsigned int &n_qubits, const unsigned int &r, const unsigned int* const j_vector);

// Loops over all sets of basis vector indices j_vector = (j_1, j_2, ..., j_r) from j_k = 1 to n_qubits for all k and evaluates operate_basis(j_vector) on each iteration
template<typename LoopFunc>
void nested_basis_loop(const unsigned int &n_qubits, const unsigned int &nested_loops, LoopFunc operate_basis);

#endif