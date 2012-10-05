/*
 * C++ification of Nikolaus Hansen's original C-source code for the
 * CMA-ES
 *
 * C++-ificiation performed by Maarten Keijzer (C) 2005. Licensed under
 * the LGPL. Original copyright of Nikolaus Hansen can be found below
 *
 *
 *
 */

/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
/* --- File: cmaes.c  -------- Author: Nikolaus Hansen   --- */
/* --------------------------------------------------------- */
/*
 *      CMA-ES for non-linear function minimization.
 *
 *           Copyright (C) 1996, 2003  Nikolaus Hansen.
 *           e-mail: hansen@bionik.tu-berlin.de
 *
 *           This library is free software; you can redistribute it and/or
 *           modify it under the terms of the GNU Lesser General Public
 *           License as published by the Free Software Foundation; either
 *           version 2.1 of the License, or (at your option) any later
 *           version (see http://www.gnu.org/copyleft/lesser.html).
 *
 *           This library is distributed in the hope that it will be useful,
 *           but WITHOUT ANY WARRANTY; without even the implied warranty of
 *           MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *           Lesser General Public License for more details.
 *
 *                                                             */
/* --- Changes : ---
 *   03/03/21: argument const double *rgFunVal of
 *   cmaes_ReestimateDistribution() was treated incorrectly.
 *   03/03/29: restart via cmaes_resume_distribution() implemented.
 *   03/03/30: Always max std dev / largest axis is printed first.
 *   03/08/30: Damping is adjusted for large mueff.
 *   03/10/30: Damping is adjusted for large mueff always.
 *   04/04/22: Cumulation time and damping for step size adjusted.
 *   No iniphase but conditional update of pc.
 *   Version 2.23.
 *                               */

#include <es/CMAParams.h>
#include <utils/eoParser.h>

#include <string>

using namespace std;

namespace eo {

CMAParams::CMAParams(eoParser& parser, unsigned dimensionality) {

    string section = "CMA parameters";

    n = parser.createParam(dimensionality, "dimensionality", "Dimensionality (N) of the problem", 'N', section, dimensionality == 0).value();

    maxgen = parser.createParam(
            1000,
            "max-gen",
            "Maximum number of generations that the system will run (needed for damping)",
            'M',
            section).value();


    if (n == 0) {
        return;
    }

    defaults(n, maxgen);

    /* handle lambda */
    lambda = parser.createParam(
            lambda,
            "lambda",
            "Number of offspring",
            'l',
            section).value();

    if (lambda < 2) {
        lambda = 4+(int)(3*log((double) n));
        cerr << "Too small lambda specified, setting it to " << lambda << endl;
    }

    /* handle mu */
    mu = parser.createParam(
            mu,
            "mu",
            "Population size",
            'm',
            section).value();

    if (mu >= lambda) {
        mu = lambda/2;
        cerr << "Mu set larger/equal to lambda, setting it to " << mu << endl;
    }

    /* handle selection weights */

    int weight_type = parser.createParam(
            0,
            "weighting",
            "Weighting scheme (for 'selection'): 0 = logarithmic, 1 = equal, 2 = linear",
            'w',
            section).value();

    switch (weight_type) {
        case 1:
            {
                for (unsigned i = 0; i < weights.size(); ++i) {
                    weights[i] = mu - i;
                }
            }
        case 2:
            {
                weights = 1.;
            }
        default :
            {
                for (unsigned i = 0; i < weights.size(); ++i) {
                    weights[i] = log(mu+1.)-log(i+1.);
                }
            }

    }

    /* Normalize weights and set mu_eff */
    double sumw = weights.sum();
    mueff =  sumw * sumw / (weights * weights).sum();
    weights /= sumw;


    /* most of the rest depends on mu_eff, so needs to be set again */

    /* set the others using Nikolaus logic. If you want to tweak, you can parameterize over these defaults */
    mucov = mueff;
    ccumsig = (mueff + 2.) / (n + mueff + 3.);
    ccumcov = 4. / (n + 4);

    double t1 = 2. / ((n+1.4142)*(n+1.4142));
    double t2 = (2.*mucov-1.) / ((n+2.)*(n+2.)+mucov);
    t2 = (t2 > 1) ? 1 : t2;
    t2 = (1./mucov) * t1 + (1.-1./mucov) * t2;

    ccov = t2;

    damp = 1 + std::max(0.3,(1.-(double)n/(double)maxgen))
              * (1+2*std::max(0.,sqrt((mueff-1.)/(n+1.))-1)) /* limit sigma increase */
                    / ccumsig;

    vector<double> mins(1,0.0);
    mins = parser.createParam(
            mins,
            "min-stdev",
            "Array of minimum stdevs, last one will apply for all remaining axes",
            0,
            section).value();

    if (mins.size() > n) mins.resize(n);

    if (mins.size()) {
        minStdevs = mins.back();
        for (unsigned i = 0; i < mins.size(); ++i) {
            minStdevs[i] = mins[i];
        }
    }

    vector<double> inits(1,0.3);
    inits = parser.createParam(
            inits,
            "init-stdev",
            "Array of initial stdevs, last one will apply for all remaining axes",
            0,
            section).value();

    if (inits.size() > n) inits.resize(n);

    if (inits.size()) {
        initialStdevs = inits.back();
        for (unsigned i = 0; i < inits.size(); ++i) {
            initialStdevs[i] = inits[i];
        }
    }

}

void CMAParams::defaults(unsigned n_, unsigned maxgen_) {
    n = n_;
    maxgen = maxgen_;

    lambda = 4+(int)(3*log((double) n));
    mu = lambda / 2;

    weights.resize(mu);

    for (unsigned i = 0; i < weights.size(); ++i) {
        weights[i] = log(mu+1.)-log(i+1.);
    }

    /* Normalize weights and set mu_eff */
    double sumw = weights.sum();
    mueff =  sumw * sumw / (weights * weights).sum();
    weights /= sumw;

    mucov = mueff;
    ccumsig *= (mueff + 2.) / (n + mueff + 3.);
    ccumcov = 4. / (n + 4);

    double t1 = 2. / ((n+1.4142)*(n+1.4142));
    double t2 = (2.*mucov-1.) / ((n+2.)*(n+2.)+mucov);
    t2 = (t2 > 1) ? 1 : t2;
    t2 = (1./mucov) * t1 + (1.-1./mucov) * t2;

    ccov = t2;

    damp = 1 + std::max(0.3,(1.-(double)n/maxgen))
              * (1+2*std::max(0.,sqrt((mueff-1.)/(n+1.))-1)) /* limit sigma increase */
                    / ccumsig;

    minStdevs.resize(n);
    minStdevs = 0.0;

    initialStdevs.resize(n);
    initialStdevs = 0.3;


}


}// namespace eo
