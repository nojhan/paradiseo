/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2013 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef _edoEstimatorBinomial_h
#define _edoEstimatorBinomial_h

#include "edoEstimator.h"
#include "edoBinomial.h"

/** An estimator for edoBinomial
 *
 * @ingroup Estimators
 * @ingroup Binomial
 */
template< class EOT, class D = edoBinomial<EOT> >
class edoEstimatorBinomial : public edoEstimator<D>
{
    public:
        /** This generic implementation makes no assumption about the underlying
         * atom type of the EOT. It can be any type that may be casted in a 
         * double as 1 or 0.
         *
         * For instance, you can use a vector<int>, but it must contains 1 or 0.
         *
         * FIXME: Partial template specializations with a conditional branching may be more generic.
         */
        D operator()( eoPop<EOT>& pop )
        {
            unsigned int popsize = pop.size();
            assert(popsize > 0);

            unsigned int dimsize = pop[0].size();
            assert(dimsize > 0);

            D probas(dimsize, 0.0);

            for (unsigned int i = 0; i < popsize; ++i) {
                for (unsigned int d = 0; d < dimsize; ++d) {
                    assert( pop[i][d] == 0 || pop[i][d] == 1 );
                    probas[d] += static_cast<double>(pop[i][d]) / popsize;
                }
            }

            return probas;
        }
};

#endif // !_edoEstimatorBinomial_h

