#ifndef MATRICES_H
#define MATRICES_H

#include <vector>

/** @ingroup Utilities
 */
class lower_triangular_matrix {

    unsigned n;
    std::vector<double> data;

    public:

    lower_triangular_matrix(unsigned n_ = 0) : n(n_), data(n * (n+1) / 2) {};

    void resize(unsigned n_) {
        n = n_;
        data.resize(n*(n+1)/2);
    }

    std::vector<double>::iterator operator[](unsigned i) { return data.begin() + i * (i+1) / 2; }
    std::vector<double>::const_iterator operator[](unsigned i) const { return data.begin() + i*(i+1)/2; }
};

/** @ingroup Utilities
 */
class square_matrix {
    unsigned n;
    std::vector<double> data;

    public:

    square_matrix(unsigned n_ = 0) : n(n_), data(n * n) {};

    void resize(unsigned n_) {
        n = n_;
        data.resize(n*n);
    }

    std::vector<double>::iterator operator[](unsigned i) { return data.begin() + i * n; }
    std::vector<double>::const_iterator operator[](unsigned i) const { return data.begin() + i*n; }
};

#endif
