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

#include "edoSampler.h"

/** Sample points in a multi-normal law defined by a mean vector, a covariance matrix, a sigma scale factor and
 * evolution paths. This is a step of the CMA-ES algorithm.
 *
 * @ingroup Samplers
 * @ingroup CMAES
 * @ingroup Adaptivenormal
 */
#ifdef WITH_EIGEN

template< class EOT, typename D = edoNormalAdaptive< EOT > >
class edoSamplerNormalAdaptive : public edoSampler< D >
{
public:
    typedef typename EOT::AtomType AtomType;

    typedef typename D::Vector Vector;
    typedef typename D::Matrix Matrix;

    edoSamplerNormalAdaptive( edoRepairer<EOT> & repairer ) 
        : edoSampler< D >( repairer)
    {}


    EOT sample( D& distrib )
    {
        unsigned int N = distrib.size();
        assert( N > 0);

        // T = vector of size elements drawn in N(0,1)
        Vector T( N );
        for ( unsigned int i = 0; i < N; ++i ) {
            T( i ) = rng.normal();
        }
        assert(T.innerSize() == N );
        assert(T.outerSize() == 1);

        // mean(N,1) + sigma * B(N,N) * ( D(N,1) .* T(N,1) )
        Vector sol = distrib.mean()
            + distrib.sigma()
            * distrib.coord_sys() * (distrib.scaling().cwiseProduct(T) ); // C * T = B * (D .* T)
        assert( sol.size() == N );
        /*Vector sol = distrib.mean() + distrib.sigma()
            * distrib.coord_sys().dot( distrib.scaling().dot( T ) );*/

        // copy in the EOT structure (more probably a vector)
        EOT solution( N );
        for( unsigned int i = 0; i < N; i++ ) {
            solution[i]= sol(i);
        }

        return solution;
    }
};
#endif // WITH_EIGEN

#endif // !_edoSamplerNormalAdaptive_h
