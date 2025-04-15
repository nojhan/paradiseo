/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoRankingCached.h
   (c) Maarten Keijzer, Marc Schoenauer, 2001

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoRankingCached_h
#define eoRankingCached_h

#include "eoPerf2Worth.h"

/**
 * @class eoRankingCached
 * @brief Cached version of eoRanking that stores precomputed values for better performance
 *
 * This class implements the same ranking algorithm as eoRanking but adds a caching layer
 * that stores frequently used values when the population size remains constant between
 * calls. This optimization is particularly useful in steady-state evolution where the
 * population size typically doesn't change between selection operations.
 *
 * The caching mechanism stores:
 * - Population size related values (pSize, pSizeMinusOne)
 * - Precomputed coefficients (alpha, beta, gamma)
 *
 * @warning This optimization should only be used when the population size remains constant
 * between calls to the operator. For dynamic population sizes, use the standard eoRanking.
 *
 * @ingroup Selectors
 */
template <class EOT>
class eoRankingCached : public eoPerf2Worth<EOT>
{
public:
    using eoPerf2Worth<EOT>::value;

    /* Ctor:
     @param _p selective pressure (in (1,2]
     @param _e exponent (1 == linear)
    */
    eoRankingCached(double _p = 2.0, double _e = 1.0) : pressure(_p), exponent(_e), cached_pSize(0)
    {
        assert(1 < pressure and pressure <= 2);
    }

    /*
     Computes the ranked fitness with caching optimization
     Fitnesses range in [m,M] where:
     - m = 2-pressure/popSize
     - M = pressure/popSize
     The progression between m and M depends on the exponent (linear when exponent=1)

     @param _pop The population to rank
     */
    virtual void operator()(const eoPop<EOT> &_pop)
    {
        unsigned pSize = _pop.size();

        if (pSize <= 1)
            throw eoPopSizeException(pSize, "cannot do ranking with population of size <= 1");

        // value() refers to the std::vector of worthes (we're in an eoParamvalue)
        value().resize(pSize);

        // Cache population-size dependent values only when population size changes
        if (pSize != cached_pSize)
        {
            cached_pSize = pSize;
            cached_pSizeMinusOne = pSize - 1;
            cached_beta = (2 - pressure) / pSize;
            cached_gamma = (2 * pressure - 2) / pSize;
            cached_alpha = (2 * pressure - 2) / (pSize * cached_pSizeMinusOne);
        }

        std::vector<const EOT *> rank;
        _pop.sort(rank);

        // map of indices for the population
        std::unordered_map<const EOT *, unsigned> indexMap;
        for (unsigned i = 0; i < pSize; ++i)
        {
            indexMap[&_pop[i]] = i;
        }

        if (exponent == 1.0) // no need for exponential then (linear case)
        {
            for (unsigned i = 0; i < pSize; i++)
            {
                const EOT *indiv = rank[i];
                int which = indexMap[indiv];
                value()[which] = cached_alpha * (pSize - i) + cached_beta;
            }
        }
        else // non-linear case (exponent != 1)
        {
            for (unsigned i = 0; i < pSize; i++)
            {
                const EOT *indiv = rank[i];
                int which = indexMap[indiv];
                // value is in [0,1]
                double tmp = ((double)(pSize - i)) / pSize;
                // to the exponent, and back to [m,M]
                value()[which] = cached_gamma * pow(tmp, exponent) + cached_beta;
            }
        }
    }

private:
    double pressure; // selective pressure (1 < pressure <= 2)
    double exponent; // exponent (1 = linear)

    // Cached values (recomputed only when population size changes)
    unsigned cached_pSize;         // last seen population size
    unsigned cached_pSizeMinusOne; // pSize - 1
    double cached_alpha;           // linear scaling coefficient
    double cached_beta;            // base value coefficient
    double cached_gamma;           // non-linear scaling coefficient
};

#endif
