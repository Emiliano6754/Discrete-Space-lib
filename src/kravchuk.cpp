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

// Returns all Kravchuk polynomials from rank 0 to rank max_rank, with N fixed
std::vector<polynomial> get_Kravchuk_pols(unsigned int const &max_rank, unsigned int const &N) {
    std::vector<polynomial> pols;
    pols.reserve(max_rank+1);
    // Calculates polynomials by thirds, to avoid expensive copying
    std::unique_ptr<int[]> first_ptr = std::make_unique<int[]>(1);
    std::unique_ptr<int[]> second_ptr = std::make_unique<int[]>(2);
    first_ptr[0] = 1;
    second_ptr[0] = N;
    second_ptr[1] = -2;
    unsigned int current_rank = 2;
    polynomial first(0, first_ptr), second(0, second_ptr);
    pols.push_back(first);
    pols.push_back(second);
    while(current_rank <= max_rank) {
        polynomial third = get_next_Kravchuk(first, second, N);
        pols.push_back(third);
        if (current_rank > max_rank) {
            break;
        }
        polynomial first = get_next_Kravchuk(second, third, N);
        pols.push_back(first);
        if (current_rank > max_rank) {
            break;
        }
        polynomial second = get_next_Kravchuk(third, first, N);
        pols.push_back(second);
    }

    return pols;
}