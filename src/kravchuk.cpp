#include "kravchuk.h"
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

// Builds a polynomial from its coefficients
polynomial3::polynomial3(unsigned int const &rank1, unsigned int const &rank2, unsigned int const &rank3, std::unique_ptr<double[]> const &in_coeffs) : n_rank1(rank1), n_rank2(rank2), n_rank3(rank3), coeffs(std::make_unique<double[]>((rank1+1)*(rank2+1)*(rank3+1))) {
    for (int j = 0; j <= rank3; j++) {
        for (int k = 0; k <= rank2; k++) {
            for (int l = 0; l <= rank1; l++) {
                (*this)(j, k, l) = in_coeffs[l + (k + j * (rank2 + 1)) * (rank1 + 1)];
            }
        }
    }
}

// Copy constructor
polynomial3::polynomial3(polynomial3 const &other): n_rank1(other.n_rank1), n_rank2(other.n_rank2), n_rank3(other.n_rank3), coeffs(std::make_unique<double[]>((other.n_rank1 + 1) * (other.n_rank2 + 1) * (other.n_rank3 + 1))) {
    std::copy(other.coeffs.get(), other.coeffs.get() + (other.n_rank1 + 1) * (other.n_rank2 + 1) * (other.n_rank3 + 1), coeffs.get());
}

// Constructs a polynomial over three variables from three polynomials, each over one variable
polynomial3::polynomial3(polynomial const &p1, polynomial const &p2, polynomial const &p3) : n_rank1(p1.rank()), n_rank2(p2.rank()), n_rank3(p3.rank()), coeffs(std::make_unique<double[]>((p1.rank()+1)*(p2.rank()+1)*(p3.rank()+1))) {
    for (int j = 0; j <= n_rank3; j++) {
        for (int k = 0; k <= n_rank2; k++) {
            for (int l = 0; l <= n_rank1; l++) {
                (*this)(j, k, l) = p1[j] * p2[k] * p3[l];
            }
        }
    }
}

// Returns this evaluated at (m,n,k)
double const polynomial3::eval(int const &m, int const &n, int const &k) const {
    double res = 0;
    double current_x = 1, current_y = 1, current_z = 1;
    for (int j = 0; j <= n_rank3; j++) {
        for (int k = 0; k <= n_rank2; k++) {
            for (int l = 0; l <= n_rank1; l++) {
                res += (*this)(j, k, l) * current_x * current_y * current_z;
                current_x *= m;
            }
            current_y *= n;
        }
        current_z *= k;
    }
    return res;
}

// Returns a reference to the (m,n,k)-th power coefficient of this
double& polynomial3::operator()(int const &m, int const &n, int const &k) {
    return coeffs[m + (n + k * (n_rank2 + 1)) * (n_rank1 + 1)];
}

// Returns a const reference to the (m,n,k)-th power coefficient of this
double const& polynomial3::operator()(int const &m, int const &n, int const &k) const {
    return coeffs[m + (n + k * (n_rank2 + 1)) * (n_rank1 + 1)];
}

// Adds other to this as polynomials. If this is smaller in any rank to other, a complete copy must be performed in order to address the new compression 
void polynomial3::operator+=(polynomial3 const &other) {
    if (other.n_rank1 <= n_rank1 && other.n_rank2 <= n_rank2 && other.n_rank3 <= n_rank3) {
        for (int j = 0; j <= other.n_rank3; j++) {
            for (int k = 0; k <= other.n_rank2; k++) {
                for (int l = 0; l <= other.n_rank1; l++) {
                    (*this)(j, k, l) += other(j, k, l);
                }
            }
        }
    } else {
        const int max_rank1 = std::max(n_rank1, other.n_rank1);
        const int max_rank2 = std::max(n_rank2, other.n_rank2);
        const int max_rank3 = std::max(n_rank3, other.n_rank3);
        std::unique_ptr<double[]> new_coeffs = std::make_unique<double[]>((max_rank1 + 1) * (max_rank2 + 1) * (max_rank3 + 1));
        unsigned int pos = 0;
        for (int j = 0; j <= max_rank3; j++) {
            for (int k = 0; k <= max_rank2; k++) {
                for (int l = 0; l <= max_rank1; l++) {
                    pos = l + (k + j * (max_rank2 + 1)) * (max_rank1 + 1);
                    new_coeffs[pos] = 0;
                    if (l <= n_rank1 && k <= n_rank2 && j <= n_rank3) {
                        new_coeffs[pos] += (*this)(j, k, l);
                    }
                    if (l <= other.n_rank1 && k <= other.n_rank2 && j <= other.n_rank3) {
                        new_coeffs[pos] += other(j, k, l);
                    }
                }
            }
        }
        coeffs = std::move(new_coeffs);
        n_rank1 = max_rank1;
        n_rank2 = max_rank2;
        n_rank3 = max_rank3;
    }
}
// Returns a new polynomial, with coefficients equal to the sum of coefficients of both this and other
polynomial3 polynomial3::operator+(polynomial3 const &other) {
    const int max_rank1 = std::max(n_rank1, other.n_rank1);
    const int max_rank2 = std::max(n_rank2, other.n_rank2);
    const int max_rank3 = std::max(n_rank3, other.n_rank3);
    std::unique_ptr<double[]> new_coeffs = std::make_unique<double[]>((max_rank1 + 1) * (max_rank2 + 1) * (max_rank3 + 1));
    unsigned int pos = 0;
    for (int j = 0; j <= max_rank3; j++) {
        for (int k = 0; k <= max_rank2; k++) {
            for (int l = 0; l <= max_rank1; l++) {
                pos = l + (k + j * (max_rank2 + 1)) * (max_rank1 + 1);
                new_coeffs[pos] = 0;
                if (l <= n_rank1 && k <= n_rank2 && j <= n_rank3) {
                    new_coeffs[pos] += coeffs[l + (k + j * (n_rank2 + 1)) * (n_rank1 + 1)];
                }
                if (l <= other.n_rank1 && k <= other.n_rank2 && j <= other.n_rank3) {
                    new_coeffs[pos] += other.coeffs[l + (k + j * (other.n_rank2 + 1)) * (other.n_rank1 + 1)];
                }
            }
        }
    }
    return polynomial3(max_rank1, max_rank2, max_rank3, new_coeffs);
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