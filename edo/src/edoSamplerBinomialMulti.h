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

#ifndef _edoSamplerBinomialMulti_h
#define _edoSamplerBinomialMulti_h

#include <utils/eoRNG.h>

#include "edoSampler.h"
#include "edoBinomialMulti.h"

#ifdef WITH_EIGEN
#include <Eigen/Dense>

/** A sampler for an edoBinomialMulti distribution.
 *
 * @ingroup Samplers
 * @ingroup Binomial
 */
template< class EOT, class D = edoBinomialMulti<EOT> >
class edoSamplerBinomialMulti : public edoSampler<D>
{
public:
    EOT sample( D& distrib )
    {
        unsigned int rows = distrib.rows();
        unsigned int cols = distrib.cols();
        assert(rows > 0);
        assert(cols > 0);

        // The point we want to draw
        // x = {x1, x2, ..., xn}
        EOT solution;

        // Sampling all dimensions
        for( unsigned int i = 0; i < rows; ++i ) {
            typename EOT::AtomType vec;
            for( unsigned int j = 0; j < cols; ++j ) {
                // Toss a coin, biased by the proba of being 1.
                vec.push_back( rng.flip( distrib(i,j) ) );
            }
            solution.push_back( vec );
        }

        return solution;
    }
};

#endif // WITH_EIGEN
#endif // !_edoSamplerBinomialMulti_h

