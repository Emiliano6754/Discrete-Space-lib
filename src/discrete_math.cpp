#include "discrete_math.h"
#include<iostream>

// Returns a double approximation of Binom(N,k)
double binom(unsigned int const &N, unsigned int const &k) {
    double res = 1;
    for (int j = 1; j <= k; j++) {
        res *= static_cast<double>(N + 1 - j) / j;
    }
    return res;
}

// Returns a buffer storing all binomial coefficients from (N, 0) to (N, N)
std::vector<double> get_binoms(unsigned int const &N) {
    std::vector<double> buffer(N + 1);
    buffer[0] = 1;
    for (unsigned int k = 0; k < N+1; k++) {
        buffer[k] = binom(N, k);
    }
    return buffer;
}

// Returns a buffer storing all squared binomial coefficients from (N, 0) to (N, N)
std::vector<double> get_binoms2(unsigned int const &N) {
    std::vector<double> buffer(N + 1);
    buffer[0] = 1;
    for (unsigned int k = 0; k < N+1; k++) {
        buffer[k] = binom(N, k);
        buffer[k] *= buffer[k];
    }
    return buffer;
}

// Returns a const reference to a cached buffer storing all binomial coefficients from (N, 0) to (N, N). Recomputes the cache whenever the introduced N differs from the last input
std::vector<double> const& cached_binoms(unsigned int const &N) {
    static unsigned int current_N = 0;
    static std::vector<double> cache;
    if (N != current_N) {
        current_N = N;
        cache.resize(N + 1);
        for (int k = 0; k <= N; k++) {
            cache[k] = binom(N, k);
        }
    }
    return cache;
}

// Parses an unsigned int inputted by the user
unsigned int parse_unsigned_int() {
    unsigned int parsed_input = 0;
    std::string input = "";
    std::cin >> input;
    try {
        unsigned long u = std::stoul(input);
        if (u > std::numeric_limits<unsigned int>::max())
            throw std::out_of_range(input);
        parsed_input = u;
    } catch (const std::invalid_argument& e) {
        std::cout << "Input could not be parsed: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "Input out of range: " << e.what() << std::endl;
    }
    return parsed_input;
}

// Asks the user with prompt to enter an unsigned int and returns the parsed answer
unsigned int ask_unsigned_int(std::string const &prompt) {
    std::cout << prompt << std::endl;
    return parse_unsigned_int();
}

// Asks the user with prompt to enter an arbitrary number of unsigned ints
std::vector<unsigned int> ask_unsigned_ints(std::string const &prompt) {
    std::vector<unsigned int> numbers;
    std::string line;
    std::cout << prompt << std::endl;
    std::cout << "Leave empty or send f to exit" << std::endl;

    if (std::cin.peek() == '\n') {
        std::cin.ignore();
    }
    while (true) {
        std::getline(std::cin, line);

        if (line.empty() || line == "f")
            break;

        std::istringstream iss(line);
        unsigned int num;
        if (iss >> num) {
            numbers.push_back(num);
        } else {
            std::cout << "Invalid input. Please enter an integer, 'f', or an empty line to finish.\n";
        }
    }
    return numbers;
}

double parse_double() {
    double parsed_input = 0;
    std::string input = "";
    std::cin >> input;
    try {
        double u = std::stod(input);
        if (u > std::numeric_limits<double>::max())
            throw std::out_of_range(input);

        parsed_input = u;
    } catch (const std::invalid_argument& e) {
        std::cout << "Input could not be parsed: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "Input out of range: " << e.what() << std::endl;
    }
    return parsed_input;
}

double ask_double(std::string const &prompt) {
    std::cout << prompt << std::endl;
    return parse_double();
}

// Asks the user with prompt (automatically tells the user that 0 means false) to enter a bool and returns the parsed answer
bool ask_bool(std::string const &prompt) {
    std::cout << prompt << std::endl;
    std::cout << "Enter 0 for false, or any other number for true" << std::endl;
    bool ans = false;
    std::cin >> ans;
    return ans;
}
