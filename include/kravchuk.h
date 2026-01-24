#ifndef KRAVCHUK_H
#define KRAVCHUK_H

#include<vector>
#include<memory>

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

//  Builds the next Kravchuk polynomial from their recurrence relations
polynomial get_next_Kravchuk(polynomial const &first, polynomial const &second, unsigned int const &N);

// Returns all Kravchuk polynomials from rank 0 to rank max_rank, with N fixed
std::vector<polynomial> get_Kravchuk_pols(unsigned int const &max_rank, unsigned int const &N);

//  Builds the next Kravchuk polynomial from their recurrence relations
polynomial get_next_Kravchuk(polynomial const &first, polynomial const &second, unsigned int const &N);

// Returns all Kravchuk polynomials from rank 0 to rank max_rank, with N fixed
std::vector<polynomial> get_Kravchuk_pols(unsigned int const &max_rank, unsigned int const &N);

#endif