/*
 * C++ification of Nikolaus Hansen's original C-source code for the
 * CMA-ES
 *
 * C++-ificiation performed by Maarten Keijzer (C) 2005. Licensed under
 * the LGPL. Original copyright of Nikolaus Hansen can be found below
 *
 *
 * Some changes have been made to the original flow and logic of the
 * algorithm:
 *
 *	- Numerical issues are now treated 'before' going into the eigenvector decomposition
 *          (this was done out of convenience)
 *	- dMaxSignifiKond (smallest x such that x == x + 1.0) replaced by
 *        numeric_limits<double>::epsilon() (smallest x such that 1.0 != 1.0 + x)
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

#include <valarray>
#include <limits>
#include <iostream>
#include <cassert>

#include <utils/eoRNG.h>

#include <es/CMAState.h>
#include <es/CMAParams.h>
#include <es/matrices.h>
#include <es/eig.h>

using namespace std;

namespace eo {

struct CMAStateImpl {

    CMAParams p;

    lower_triangular_matrix	C; // Covariance matrix
    square_matrix		B; // Eigen vectors (in columns)
    valarray<double>		d; // eigen values (diagonal matrix)
    valarray<double>		pc; // Evolution path
    valarray<double>		ps; // Evolution path for stepsize;

    vector<double>		mean; // current mean to sample around
    double			sigma; // global step size

    unsigned			gen;
    vector<double>		fitnessHistory;


    CMAStateImpl(const CMAParams& params_, const vector<double>& m, double sigma_) :
        p(params_),
        C(p.n), B(p.n), d(p.n), pc(p.n), ps(p.n), mean(m), sigma(sigma_),
        gen(0), fitnessHistory(3)
    {
        double trace = (p.initialStdevs * p.initialStdevs).sum();
        /* Initialize covariance structure */
        for (unsigned i = 0; i < p.n; ++i)
        {
            B[i][i] = 1.;
            d[i] = p.initialStdevs[i] * sqrt(p.n / trace);
            C[i][i] = d[i] * d[i];
            pc[i] = 0.;
            ps[i] = 0.;
        }

    }

    void sample(vector<double>& v) {
        unsigned n = p.n;
        v.resize(n);

        vector<double> tmp(n);
        for (unsigned i = 0; i < n; ++i)
            tmp[i] = d[i] * rng.normal();

        /* add mutation (sigma * B * (D*z)) */
        for (unsigned i = 0; i < n; ++i) {
            double sum = 0;
            for (unsigned j = 0; j < n; ++j) {
                sum += B[i][j] * tmp[j];
            }
            v[i] = mean[i] + sigma * sum;
        }
    }

    void reestimate(const vector<const vector<double>* >& pop, double muBest, double muWorst) {

        assert(pop.size() == p.mu);

        unsigned n = p.n;

        fitnessHistory[gen % fitnessHistory.size()] = muBest; // needed for divergence check

        vector<double> oldmean = mean;
        valarray<double> BDz(n);

        /* calculate xmean and rgBDz~N(0,C) */
        for (unsigned i = 0; i < n; ++i) {
            mean[i] = 0.;
            for (unsigned j = 0; j < pop.size(); ++j) {
                mean[i] += p.weights[j] * (*pop[j])[i];
            }
            BDz[i] = sqrt(p.mueff)*(mean[i] - oldmean[i])/sigma;
        }

        vector<double> tmp(n);
        /* calculate z := D^(-1) * B^(-1) * rgBDz into rgdTmp */
        for (unsigned i = 0; i < n; ++i) {
            double sum = 0.0;
            for (unsigned j = 0; j < n; ++j) {
                sum += B[j][i] * BDz[j];
            }
            tmp[i] = sum / d[i];
        }

        /* cumulation for sigma (ps) using B*z */
        for (unsigned i = 0; i < n; ++i) {
            double sum = 0.0;
            for (unsigned j = 0; j < n; ++j)
                sum += B[i][j] * tmp[j];

            ps[i] = (1. - p.ccumsig) * ps[i] + sqrt(p.ccumsig * (2. - p.ccumsig)) * sum;
        }

        /* calculate norm(ps)^2 */
        double psxps = (ps * ps).sum();


        double chiN =  sqrt((double) p.n) * (1. - 1./(4.*p.n) + 1./(21.*p.n*p.n));
        /* cumulation for covariance matrix (pc) using B*D*z~N(0,C) */
        double hsig = sqrt(psxps) / sqrt(1. - pow(1.-p.ccumsig, 2.*gen)) / chiN < 1.5 + 1./(p.n-0.5);

        pc = (1. - p.ccumcov) * pc + hsig * sqrt(p.ccumcov * (2. - p.ccumcov)) * BDz;

        /* stop initial phase (MK, this was not reachable in the org code, deleted) */

        /* remove momentum in ps, if ps is large and fitness is getting worse */

        if (gen >= fitnessHistory.size()) {

            // find direction from muBest and muWorst (muBest == muWorst handled seperately
            double direction = muBest < muWorst? -1.0 : 1.0;

            unsigned now = gen % fitnessHistory.size();
            unsigned prev = (gen-1) % fitnessHistory.size();
            unsigned prevprev = (gen-2) % fitnessHistory.size();

            bool fitnessWorsens = (muBest == muWorst) || // <- increase norm also when population has converged (this deviates from Hansen's scheme)
                            ( (direction * fitnessHistory[now] < direction * fitnessHistory[prev])
                                            &&
                              (direction * fitnessHistory[now] < direction * fitnessHistory[prevprev]));

            if(psxps/p.n > 1.5 + 10.*sqrt(2./p.n) && fitnessWorsens) {
                double tfac = sqrt((1 + std::max(0., log(psxps/p.n))) * p.n / psxps);
                ps          *= tfac;
                psxps   *= tfac*tfac;
            }
        }

        /* update of C  */
        /* Adapt_C(t); not used anymore */
        if (p.ccov != 0.) {
            //flgEigensysIsUptodate = 0;

            /* update covariance matrix */
            for (unsigned i = 0; i < n; ++i) {
                vector<double>::iterator c_row = C[i];
                for (unsigned j = 0; j <= i; ++j) {
                    c_row[j] =
                                (1 - p.ccov) * c_row[j]
                                        +
                                p.ccov * (1./p.mucov) * pc[i] * pc[j]
                                        +
                                (1-hsig) * p.ccumcov * (2. - p.ccumcov) * c_row[j];

                    /*C[i][j] = (1 - p.ccov) * C[i][j]
                        + sp.ccov * (1./sp.mucov)
                        * (rgpc[i] * rgpc[j]
                                + (1-hsig)*sp.ccumcov*(2.-sp.ccumcov) * C[i][j]); */
                    for (unsigned k = 0; k < p.mu; ++k) { /* additional rank mu update */
                        c_row[j] += p.ccov * (1-1./p.mucov) * p.weights[k]
                            * ( (*pop[k])[i] - oldmean[i])
                            * ( (*pop[k])[j] - oldmean[j])
                            / sigma / sigma;

                           // * (rgrgx[index[k]][i] - rgxold[i])
                           // * (rgrgx[index[k]][j] - rgxold[j])
                           // / sigma / sigma;
                    }
                }
            }
        }

        /* update of sigma */
        sigma *= exp(((sqrt(psxps)/chiN)-1.)/p.damp);
        /* calculate eigensystem, must be done by caller  */
        //cmaes_UpdateEigensystem(0);


        /* treat minimal standard deviations and numeric problems
         * Note that in contrast with the original code, some numerical issues are treated *before* we
         * go into the eigenvalue calculation */

        treatNumericalIssues(muBest, muWorst);

        gen++; // increase generation
    }

    void treatNumericalIssues(double best, double worst) {

        /* treat stdevs */
        for (unsigned i = 0; i < p.n; ++i) {
            if (sigma * sqrt(C[i][i]) < p.minStdevs[i]) {
                // increase stdev
                sigma *= exp(0.05+1./p.damp);
                break;
            }
        }

        /* treat convergence */
        if (best == worst) {
            sigma *= exp(0.2 + 1./p.damp);
        }

        /* Jede Hauptachse i testen, ob x == x + 0.1 * sigma * rgD[i] * B[i] */
        /* Test if all the means are not numerically out of whack with our coordinate system*/
        for (unsigned axis = 0; axis < p.n; ++axis) {
            double fac = 0.1 * sigma * d[axis];
            unsigned coord;
            for (coord = 0; coord < p.n; ++coord) {
                if (mean[coord] != mean[coord] + fac * B[coord][axis]) {
                    break;
                }
            }

            if (coord == p.n) { // means are way too big (little) for numerical accuraccy. Start rocking the craddle a bit more
                sigma *= exp(0.2+1./p.damp);
            }

        }

        /* Testen ob eine Komponente des Objektparameters festhaengt */
        /* Correct issues with scale between objective values and covariances */
        bool theresAnIssue = false;

        for (unsigned i = 0; i < p.n; ++i) {
            if (mean[i] == mean[i] + 0.2 * sigma * sqrt(C[i][i])) {
                C[i][i] *= (1. + p.ccov);
                theresAnIssue = true;
            }
        }

        if (theresAnIssue) {
            sigma *= exp(0.05 + 1./p.damp);
        }
    }


    bool updateEigenSystem(unsigned max_tries, unsigned max_iters) {

        if (max_iters==0) max_iters = 30 * p.n;

        static double lastGoodMinimumEigenValue = 1.0;

        /* Try to get a valid calculation */
        for (unsigned tries = 0; tries < max_tries; ++tries) {

            unsigned iters = eig( p.n, C, d, B, max_iters);
            if (iters < max_iters)
            { // all is well

                /* find largest/smallest eigenvalues */
                double minEV = d.min();
                double maxEV = d.max();

                /* (MK Original comment was) :Limit Condition of C to dMaxSignifKond+1
                 * replaced dMaxSignifKond with 1./numeric_limits<double>::epsilon()
                 * */
                if (maxEV * numeric_limits<double>::epsilon() > minEV) {
                    double tmp = maxEV * numeric_limits<double>::epsilon() - minEV;
                    minEV += tmp;
                    for (unsigned i=0;i<p.n;++i) {
                        C[i][i] += tmp;
                        d[i] += tmp;
                    }
                } /* if */
                lastGoodMinimumEigenValue = minEV;

                d = sqrt(d);

                //flgEigensysIsUptodate = 1;
                //genOfEigensysUpdate = gen;
                //clockeigensum += clock() - clockeigenbegin;
                return true;
            } /* if cIterEig < ... */

            // numerical problems, ignore them and try again

            /* Addition des letzten minEW auf die Diagonale von C */
            /* Add the last known good eigen value to the diagonal of C */
            double summand = lastGoodMinimumEigenValue * exp((double) tries);
            for (unsigned i = 0; i < p.n; ++i)
                C[i][i] += summand;

        } /* for iEigenCalcVers */

        return false;

    }


};

CMAState::CMAState(const CMAParams& params, const std::vector<double>& initial_point, const double initial_sigma)
    : pimpl(new CMAStateImpl(params, initial_point, initial_sigma)) {}

CMAState::~CMAState() { delete pimpl; }
CMAState::CMAState(const CMAState& that) : pimpl(new CMAStateImpl(*that.pimpl )) {}
CMAState& CMAState::operator=(const CMAState& that) { *pimpl = *that.pimpl; return *this; }

void CMAState::sample(vector<double>& v) const {  pimpl->sample(v); }

void CMAState::reestimate(const vector<const vector<double>* >& population, double muBest, double muWorst) { pimpl->reestimate(population, muBest, muWorst); }
bool CMAState::updateEigenSystem(unsigned max_tries, unsigned max_iters) { return pimpl->updateEigenSystem(max_tries, max_iters); }


} // namespace eo
