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
    Pierre Savéant <pierre.saveant@thalesgroup.com>
*/

#ifndef _edoSamplerNormalAdaptive_h
#define _edoSamplerNormalAdaptive_h

#include <cmath>
#include <limits>

#include <edoSampler.h>

/** Sample points in a multi-normal law defined by a mean vector and a covariance matrix.
 *
 * Given M the mean vector and V the covariance matrix, of order n:
 *   - draw a vector T in N(0,I) (i.e. each value is drawn in a normal law with mean=0 an stddev=1)
 *   - compute the Cholesky decomposition L of V (i.e. such as V=LL*)
 *   - return X = M + LT
 */
#ifdef WITH_EIGEN

template< class EOT, typename EOD = edoNormalAdaptive< EOT > >
class edoSamplerNormalAdaptive : public edoSampler< EOD >
{
public:
    typedef typename EOT::AtomType AtomType;

    typedef typename EOD::Vector Vector;
    typedef typename EOD::Matrix Matrix;

    edoSamplerNormalAdaptive( edoRepairer<EOT> & repairer ) 
        : edoSampler< EOD >( repairer)
    {}


    EOT sample( EOD& distrib )
    {
        unsigned int size = distrib.size();
        assert(size > 0);

        // T = vector of size elements drawn in N(0,1)
        Vector T( size );
        for ( unsigned int i = 0; i < size; ++i ) {
            T( i ) = rng.normal();
        }
        assert(T.innerSize() == size);
        assert(T.outerSize() == 1);

        Vector sol = distrib.mean() + distrib.sigma() * distrib.coord_sys() * (distrib.scaling().dot(T) );
        /*Vector sol = distrib.mean() + distrib.sigma()
            * distrib.coord_sys().dot( distrib.scaling().dot( T ) );*/

        // copy in the EOT structure (more probably a vector)
        EOT solution( size );
        for( unsigned int i = 0; i < size; i++ ) {
            solution[i]= sol(i);
        }

        return solution;
    }
};
#endif // WITH_EIGEN

#endif // !_edoSamplerNormalAdaptive_h
