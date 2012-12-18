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

#ifndef _edoSampler_h
#define _edoSampler_h

#include <eoFunctor.h>

#include "edoRepairer.h"
#include "edoBounderNo.h"


/** @defgroup Samplers
 *
 * Functors that draw and repair individuals according to a given distribution.
 */

/** Base class for samplers
 *
 * The functor here is already implemented: it first sample an EOT from the
 * given distribution, and then apply the given repairers (if set).
 *
 * Thus, the function that need to be overloaded is "sample", unlike most of EO
 * functors.
 *
 * @ingroup Samplers
 * @ingroup Core
 */
template < typename D >
class edoSampler : public eoUF< D&, typename D::EOType >
{
public:
    typedef typename D::EOType EOType;

    edoSampler(edoRepairer< EOType > & repairer)
        : _dummy_repairer(), _repairer(repairer)
    {}

    
    edoSampler()
        : _dummy_repairer(), _repairer( _dummy_repairer )
    {}
    

    // virtual EOType operator()( D& ) = 0 (provided by eoUF< A1, R >)

    EOType operator()( D& distrib )
    {
        assert( distrib.size() > 0 );

        // Point we want to sample to get higher a set of points
        // (coordinates in n dimension)
        // x = {x1, x2, ..., xn}
        // the sample method is implemented in the derivated class
        EOType solution(sample(distrib));

        // Now we are bounding the distribution thanks to min and max
        // parameters.
        _repairer(solution);

        return solution;
    }

protected:

    virtual EOType sample( D& ) = 0;

private:
    edoBounderNo<EOType> _dummy_repairer;

    //! repairer functor
    edoRepairer< EOType > & _repairer;

};

#endif // !_edoSampler_h
