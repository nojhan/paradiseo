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
class edoEstimatorBinomial : public edoEstimator< edoBinomial< EOT > >
{
    public:
        /** This generic implementation makes no assumption about the underlying
         * atom type of the EOT. It can be any type that may be interpreted as
         * true or false.
         *
         * For instance, you can use a vector<int>, and any variable greater
         * than one will increase the associated probability.
         *
         * FIXME: Partial template specializations without the conditional branching may be more efficient.
         */
        edoBinomial<EOT> operator()( eoPop<EOT>& pop )
        {
            unsigned int popsize = pop.size();
            assert(popsize > 0);

            unsigned int dimsize = pop[0].size();
            assert(dimsize > 0);

            edoBinomial<EOT> probas(dimsize, 0.0);

            for (unsigned int i = 0; i < popsize; ++i) {
                for (unsigned int d = 0; d < dimsize; ++d) {
                    // if this variable is true (whatever that means, x=1, x=true, etc.)
                    // if( pop[i][d] ) {
                    //     probas[d] += 1/popsize; // we hope the compiler optimization pre-computes 1/p
                    // }
                    probas[d] += static_cast<double>(pop[i][d]) / popsize;
                }
            }

            return probas;
        }
};

#endif // !_edoEstimatorBinomial_h

