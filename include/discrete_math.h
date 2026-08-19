#ifndef DISCRETE_MATH_H
#define DISCRETE_MATH_H

#include<vector>
#include<functional>
#include<unsupported/Eigen/CXX11/Tensor>

constexpr double SQRT3 = 1.73205080756887;

// Lambda that converts all negative numbers to zero
inline constexpr auto negatives_to_zero = [](double x) {
    return (x < 0) ? 0.0 : x;
};

// Lambda that converts all zeros to one. Useful when dividing a symmetric function by another
inline constexpr auto zeros_to_one = [](double x) {
    return (x == 0) ? 1 : x;
};

// Returns the j-th bit of string, from right to left
inline unsigned int get_bit(unsigned int const &string, int const &j) {
    return ((string >> j) & 1u);
}

// Returns the j-th bit of string, from right to left
inline unsigned int get_bit(unsigned long long const &string, int const &j) {
    return ((string >> j) & 1u);
}

// Calculates the field-wise trace of alpha by calculating its hamming weight and returning the last bit (modulo 2)
inline int trace(unsigned int const &alpha) {
    return std::popcount(alpha) & 1;
}

// Calculates the trace of the product by doing bitwise and. Equivalent to calling trace(alpha&beta)
inline int trace(unsigned int const &alpha, unsigned int const &beta) {
    return std::popcount(alpha & beta) & 1;
}

// Returns (-1)^a
inline double sign(unsigned int const &a) {
    return 1.0 - 2.0 * (a & 1);
}

// Returns (-1)^(a+b)
inline double sign(unsigned int const &a, unsigned int const &b) {
    return 1.0 - 2.0 * ( (a + b) & 1);
}

// Returns a buffer storing all powers of var from 0 to max_power
template<typename T>
std::vector<T> power_buffer(T const &var, unsigned int const &max_power) {
    std::vector<T> buffer(max_power + 1);
    buffer[0] = 1;
    for (unsigned int j = 1; j <= max_power; j++) {
        buffer[j] = buffer[j - 1] * var;
    }
    return buffer;
}

// Returns a const reference to a cached buffer storing all powers of var from 0 to max_power. Recomputes the cache whenever the inputs differs from the last inputs
template<typename T>
std::vector<T> const& cached_power_buffer(T const &var, unsigned int const &max_power) {
    static std::vector<T> buffer(max_power + 1);
    static T current_var = 1;
    static unsigned int current_max_power = 0;

    if (var != current_var) {
        current_var = var;
        buffer.clear();
        buffer.reserve(max_power + 1);
        buffer.push_back(1);
        current_max_power = 0;
    }
    if (max_power != current_max_power) {
        buffer.reserve(max_power + 1);
        while (max_power > current_max_power) {
            buffer.push_back(buffer[buffer.size() - 1] * var);
            current_max_power++;
        }
    }
    return buffer;
}


// Returns a double approximation of Binom(N,k)
double binom(unsigned int const &N, unsigned int const &k);

// Returns a buffer storing all binomial coefficients from (N, 0) to (N, N)
std::vector<double> get_binoms(unsigned int const &N);

// Returns a buffer storing all squared binomial coefficients from (N, 0) to (N, N)
std::vector<double> get_binoms2(unsigned int const &N);

// Returns a const reference to a cached buffer storing all binomial coefficients from (N, 0) to (N, N). Recomputes the cache whenever the introduced N differs from the last input
std::vector<double> const& cached_binoms(unsigned int const &N);

// Parses an unsigned int inputted by the user
unsigned int parse_unsigned_int();

// Asks the user with prompt to enter an unsigned int and returns the parsed answer
unsigned int ask_unsigned_int(std::string const &prompt);

// Asks the user with prompt to enter an arbitrary number of unsigned ints
std::vector<unsigned int> ask_unsigned_ints(std::string const &prompt);

// Parses a double inputted by the user
double parse_double();

// Asks the user with prompt to enter a double and returns the parsed answer
double ask_double(std::string const &prompt);

// Asks the user with prompt (automatically tells the user that 0 means false) to enter a bool and returns the parsed answer
bool ask_bool(std::string const &prompt);

#endif