#include "Kravchuk.h"
#include<iostream>

// Builds a polynomial based on its coefficients
polynomial::polynomial(unsigned int const &rank, std::unique_ptr<double[]> const &in_coeffs): n_rank(rank), coeffs(std::make_unique<double[]>(rank+1)) {
    for (int j = 0; j <= rank; j++) {
        coeffs[j] = in_coeffs[j];
    }
}

// Copy constructor
polynomial::polynomial(polynomial const &other): n_rank(other.n_rank), coeffs(std::make_unique<double[]>(other.n_rank + 1)) {
    std::copy(other.coeffs.get(), other.coeffs.get() + (n_rank + 1), coeffs.get());
}

polynomial::polynomial(polynomial const &first, polynomial const &second, int const &N): n_rank(second.rank() + 1), coeffs(std::make_unique<double[]>(n_rank + 1)) {
    coeffs[0] = (N * second[0] - (N - static_cast<int>(n_rank) + 2) * first[0]) / static_cast<int>(n_rank);
    for (int j = 1; j <= first.rank(); j++) {
        coeffs[j] = (N * second[j] - 2 * second[j-1] - (N - static_cast<int>(n_rank) + 2) * first[j]) / static_cast<int>(n_rank);
    }
    coeffs[second.rank()] = (N * second[second.rank()] - 2 * second[second.rank() - 1]) / static_cast<int>(n_rank);
    coeffs[n_rank] = - 2 * second[second.rank()] / static_cast<int>(n_rank);
}

// Returns n-th coefficient of this
double polynomial::operator[](unsigned int const &n) const {
    return coeffs[n];
}

// The rank of this
unsigned int polynomial::rank() const {
    return n_rank;
}

// Returns this evaluated at x
double polynomial::operator()(int const &x) const {
    double result = coeffs[0];
    double current_x = x;
    for (int j = 1; j <= n_rank; j++) {
        result += coeffs[j] * current_x;
        current_x *= x;
    }
    return result;
}

void polynomial::print() const {
    for (int j = 0; j <= n_rank; j++) {
        std::cout << coeffs[j] << "x^" << j << " + ";
    }
    std::cout << "\n";
}

//  Builds the next Kravchuk polynomial from their recurrence relations
polynomial get_next_Kravchuk(polynomial const &first, polynomial const &second, unsigned int const &N) {
    const int next_rank = second.rank() + 1;
    std::unique_ptr<double[]> coeffs = std::make_unique<double[]>(next_rank);
    coeffs[0] = (N * second[0] - (N - next_rank + 2) * first[0]) / next_rank;
    for (int j = 1; j <= first.rank(); j++) {
        coeffs[j] = (N * second[j] - 2 * second[j-1] - (N - next_rank + 2) * first[j]) / next_rank;
    }
    coeffs[second.rank()] = (N * second[second.rank()] - 2 * second[second.rank() - 1]) / next_rank;
    coeffs[next_rank] = - 2 * second[second.rank()] / next_rank;
    return polynomial(next_rank, coeffs);
}

//  Builds the next Kravchuk polynomial from their recurrence relations. Coefficients only
std::unique_ptr<double[]> get_next_Kravchuk(std::unique_ptr<double[]> const &first, std::unique_ptr<double[]> const &second, unsigned int const &next_rank, unsigned int const &N) {
    std::unique_ptr<double[]> coeffs = std::make_unique<double[]>(next_rank);
    coeffs[0] = (N * second[0] - (N - next_rank + 2) * first[0]) / next_rank;
    for (int j = 1; j <= next_rank-2; j++) {
        coeffs[j] = (N * second[j] - 2 * second[j-1] - (N - next_rank + 2) * first[j]) / next_rank;
    }
    coeffs[next_rank-1] = (N * second[next_rank-1] - 2 * second[next_rank-2]) / next_rank;
    coeffs[next_rank] = - 2 * second[next_rank-1] / next_rank;
    return coeffs;
}

// Returns all Kravchuk polynomials from rank 0 to rank max_rank, with N fixed. Returns at least 2 for optimization
std::vector<polynomial> get_Kravchuk_pols(unsigned int const &max_rank, unsigned int const &N) {
    std::vector<polynomial> pols;
    pols.reserve(max_rank+1);
    std::unique_ptr<double[]> first = std::make_unique<double[]>(max_rank);
    std::unique_ptr<double[]> second = std::make_unique<double[]>(max_rank);
    // Set all values to 0 first to avoid any possible issues
    for (int j = 0; j <= max_rank; j++) {
        first[0] = second[0] = 0;
    }
    first[0] = 1;
    second[0] = N;
    second[1] = -2;
    pols.emplace_back(0, first);
    pols.emplace_back(1, second);
    unsigned int current_rank = 2;
    while(current_rank <= max_rank) {
        pols.emplace_back(pols[current_rank - 2], pols[current_rank - 1], N);
        current_rank++;
    }
    return pols;
}