#include "Kravchuk.h"

// Builds a polynomial based on its coefficients
polynomial::polynomial(unsigned int const &rank, std::unique_ptr<int[]> const &in_coeffs): n_rank(rank), coeffs(std::make_unique<int[]>(rank+1)) {
    for (int j = 0; j <= rank; j++) {
        coeffs[j] = in_coeffs[j];
    }
}

// Copy constructor
polynomial::polynomial(polynomial const &other): n_rank(other.n_rank), coeffs(std::make_unique<int[]>(other.n_rank + 1)) {
    std::copy(other.coeffs.get(), other.coeffs.get() + (n_rank + 1), coeffs.get());
}

// Returns n-th coefficient of this
int polynomial::operator[](unsigned int const &n) const {
    return coeffs[n];
}

// The rank of this
unsigned int polynomial::rank() const {
    return n_rank;
}

// Returns this evaluated at x
int polynomial::operator()(int const &x) const {
    int result = coeffs[0];
    int current_x = x;
    for (int j = 1; j <= n_rank; j++) {
        result += coeffs[j] * current_x;
        current_x *= x;
    }
    return result;
}

//  Builds the next Kravchuk polynomial from their recurrence relations
polynomial get_next_Kravchuk(polynomial const &first, polynomial const &second, unsigned int const &N) {
    const int next_rank = second.rank() + 1;
    std::unique_ptr<int[]> coeffs = std::make_unique<int[]>(next_rank);
    for (int j = 0; j <= next_rank; j++) {
        coeffs[j] = (N * first[j] - 2 * first[j-1] - (N - next_rank + 2) * second[j]) / next_rank;
    }
    return polynomial(next_rank, coeffs);
}

//  Builds the next Kravchuk polynomial from their recurrence relations. Coefficients only
std::unique_ptr<int[]> get_next_Kravchuk(std::unique_ptr<int[]> const &first, std::unique_ptr<int[]> const &second, unsigned int const &next_rank, unsigned int const &N) {
    std::unique_ptr<int[]> coeffs = std::make_unique<int[]>(next_rank);
    for (int j = 0; j <= next_rank; j++) {
        coeffs[j] = (N * first[j] - 2 * first[j-1] - (N - next_rank + 2) * second[j]) / next_rank;
    }
    return coeffs;
}

// Returns all Kravchuk polynomials from rank 0 to rank max_rank, with N fixed. Returns at least 2 for optimization
std::vector<polynomial> get_Kravchuk_pols(unsigned int const &max_rank, unsigned int const &N) {
    std::vector<polynomial> pols;
    pols.reserve(max_rank+1);
    std::unique_ptr<int[]> first = std::make_unique<int[]>(max_rank);
    std::unique_ptr<int[]> second = std::make_unique<int[]>(max_rank);
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
        current_rank++;
        pols.emplace_back(pols[pols.size() - 2], pols[pols.size() - 1], N);
    }

    return pols;
}