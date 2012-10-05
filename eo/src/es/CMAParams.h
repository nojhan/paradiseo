/*
 * C++ification of Nikolaus Hansen's original C-source code for the
 * CMA-ES.
 *
 * Copyright (C) 1996, 2003, Nikolaus Hansen
 *           (C) 2005, Maarten Keijzer
 *
 * License: LGPL
 *
 */

#ifndef CMAPARAMS_H__
#define CMAPARAMS_H__

#include <valarray>

class eoParser;
namespace eo {

class CMAParams {

    public:

    CMAParams() { /* Call this and all values need to be set by hand */ }
    CMAParams(eoParser& parser, unsigned dimensionality = 0); // 0 dimensionality -> user needs to set it

    void defaults(unsigned n_, unsigned maxgen_); /* apply all defaults using n and maxgen */

    unsigned n;
    unsigned maxgen;

    unsigned lambda;          /* -> mu */
    unsigned mu;              /* -> weights, lambda */

    std::valarray<double> weights;     /* <- mu, -> mueff -> mucov -> ccov */
    double mueff;	/* <- weights */

    double mucov;

    double damp;         /* <- ccumsig, maxeval, lambda */
    double ccumsig;      /* -> damp, <- N */
    double ccumcov;
    double ccov;         /* <- mucov, N */

    std::valarray<double> minStdevs;     /* Minimum standard deviations per coordinate (default = 0.0) */
    std::valarray<double> initialStdevs; /* Initial standard deviations per coordinate (default = 0.3) */
};

} // namespace eo

#endif
