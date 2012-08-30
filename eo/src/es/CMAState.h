/*
 * C++ification of Nikolaus Hansen's original C-source code for the
 * CMA-ES.
 *
 * Copyright (C) 1996, 2003, Nikolaus Hansen
 *           (C) 2005, Maarten Keijzer
 *
 * License: LGPL (see source file)
 *
 */

#ifndef CMASTATE_H_
#define CMASTATE_H_

#include <vector>
#include <valarray>

namespace eo {


class CMAStateImpl;
class CMAParams;
class CMAState {

    CMAStateImpl* pimpl; /* pointer to implementation, hidden in source file */

    public:

    CMAState(const CMAParams&, const std::vector<double>& initial_point, const double initial_sigma = 1.0);
    ~CMAState();
    CMAState(const CMAState&);
    CMAState& operator=(const CMAState&);

    /**
     *	sample a vector from the distribution
     *
     *   If the sample is not to your liking (i.e., not within bounds)
     *   you can do one of two things:
     *
     *   a) Call sample again
     *   b) multiply the entire vector with a number between -1 and 1
     *
     *   Do not modify the sample in any other way as this will invalidate the
     *   internal consistency of the system.
     *
     *   A final approach is to copy the sample and modify the copy externally (in the evaluation function)
     *   and possibly add a penalty depending on the size of the modification.
     *
     */
    void sample(std::vector<double>& v) const;

    /**
     * Reestimate covariance matrix and other internal parameters
     * Does NOT update the eigen system (call that seperately)
     *
     * Needs a population of mu individuals, sorted on fitness, plus
     *
     * muBest:   the best fitness in the population
     * muWorst:  the worst fitness in the population
     */

    void reestimate(const std::vector<const std::vector<double>* >& sorted_population, double muBest, double muWorst);

    /**
     * call this function after reestimate in order to update the eigen system
     * It is a seperate call to allow the user to periodically skip this expensive step
     *
     * max_iters = 0 implies 30 * N iterations
     *
     * If after max_tries still no numerically sound eigen system is constructed,
     * the function returns false
     */
    bool updateEigenSystem(unsigned max_tries = 1, unsigned max_iters = 0);
};

} // namespace eo

#endif
