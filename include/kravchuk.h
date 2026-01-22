#ifndef KRAVCHUK_H
#define KRAVCHUK_H

#include<vector>
#include<memory>

class polynomial{
public:
    // Builds a polynomial based on its coefficients
    polynomial(unsigned int const &rank, std::unique_ptr<int[]> const &in_coeffs);
    // Copy constructor
    polynomial(polynomial const &other);
    //  Builds the next Kravchuk polynomial from Kravchuk recurrence relations
    polynomial(polynomial const &first, polynomial const &second, unsigned int const &N): n_rank(second.rank() + 1), coeffs(std::make_unique<int[]>(n_rank + 1)) {
        for (int j = 0; j <= n_rank; j++) {
            coeffs[j] = (N * first[j] - 2 * first[j-1] - (N - n_rank + 2) * second[j]) / n_rank;
        }
    }
    // Returns n-th coefficient of this
    int operator[](unsigned int const &n) const;
    // The rank of this
    unsigned int rank() const;
    // Returns this evaluated at x
    int operator()(int const &x) const;
private:
    unsigned int const n_rank;
    std::unique_ptr<int[]> coeffs;
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