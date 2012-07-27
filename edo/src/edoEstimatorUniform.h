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

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
    Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _edoEstimatorUniform_h
#define _edoEstimatorUniform_h

#include "edoEstimator.h"
#include "edoUniform.h"

/** An estimator for edoUniform
 *
 * @ingroup Estimators
 */
template < typename EOT >
class edoEstimatorUniform : public edoEstimator< edoUniform< EOT > >
{
public:
    edoUniform< EOT > operator()(eoPop<EOT>& pop)
    {
        unsigned int size = pop.size();

        assert(size > 0);

        EOT min = pop[0];
        EOT max = pop[0];

        for (unsigned int i = 1; i < size; ++i)
        {
            unsigned int size = pop[i].size();

            assert(size > 0);

            // possibilité d'utiliser std::min_element et std::max_element mais exige 2 pass au lieu d'1.

            for (unsigned int d = 0; d < size; ++d)
            {
                if (pop[i][d] < min[d])
                    min[d] = pop[i][d];

                if (pop[i][d] > max[d])
                    max[d] = pop[i][d];
            }
        }

        return edoUniform< EOT >(min, max);
    }
};

#endif // !_edoEstimatorUniform_h
