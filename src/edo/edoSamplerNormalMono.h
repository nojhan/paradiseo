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

#ifndef _edoSamplerNormalMono_h
#define _edoSamplerNormalMono_h

#include <cmath>

#include "../eo/utils/eoRNG.h"

#include "edoSampler.h"
#include "edoNormalMono.h"
#include "edoBounder.h"

/** A sampler for edoNormalMono
 *
 * @ingroup Samplers
 * @ingroup Mononormal
 */
template < typename EOT, typename D = edoNormalMono< EOT > >
class edoSamplerNormalMono : public edoSampler< D >
{
public:
    typedef typename EOT::AtomType AtomType;

    edoSamplerNormalMono( edoRepairer<EOT> & repairer ) : edoSampler< D >( repairer) {}

    EOT sample( edoNormalMono<EOT>& distrib )
    {
        unsigned int size = distrib.size();
        assert(size > 0);

        // The point we want to draw
        // (coordinates in n dimension)
        // x = {x1, x2, ..., xn}
        EOT solution;
        
        // Sampling all dimensions
        for (unsigned int i = 0; i < size; ++i) {
            AtomType mean = distrib.mean()[i];
            AtomType variance = distrib.variance()[i];
            // should use the standard deviation, which have the same scale than the mean
            AtomType random = rng.normal(mean, sqrt(variance) );

            solution.push_back(random);
        }

        return solution;
    }
};

#endif // !_edoSamplerNormalMono_h
