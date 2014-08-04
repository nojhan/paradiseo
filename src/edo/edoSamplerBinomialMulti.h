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

#include "../eo/utils/eoRNG.h"

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
    typedef typename EOT::AtomType AtomType;

    /** Called if the sampler draw the item at (i,j)
     *
     * The default implementation is to push back a true boolean.
     * If you have a more complex data structure, you can just overload this.
     */
    virtual void make_true( AtomType & atom, unsigned int i, unsigned int j )
    {
        atom.push_back( 1 );
    }

    /** @see make_true */
    virtual void make_false( AtomType & atom, unsigned int i, unsigned int j )
    {
        atom.push_back( 0 );
    }

    EOT sample( D& distrib )
    {
        unsigned int rows = distrib.rows();
        unsigned int cols = distrib.cols();
        assert(rows > 0);
        assert(cols > 0);

        // The point we want to draw
        // X = {x1, x2, ..., xn}
        // with xn a container of booleans
        EOT solution;

        // Sampling all dimensions
        for( unsigned int i = 0; i < rows; ++i ) {
            AtomType atom;
            for( unsigned int j = 0; j < cols; ++j ) {
                // Toss a coin, biased by the proba of being 1.
                if( rng.flip( distrib(i,j) ) ) {
                    make_true( atom, i, j );
                } else {
                    make_false( atom, i, j );
                }
            }
            solution.push_back( atom );
        }

        return solution;
    }
};

#endif // WITH_EIGEN
#endif // !_edoSamplerBinomialMulti_h

