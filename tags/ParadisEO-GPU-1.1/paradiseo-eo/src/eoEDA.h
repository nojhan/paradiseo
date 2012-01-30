// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEDA.h
// (c) Marc Schoenauer, Maarten Keijzer, 2001
/*
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

    Contact: Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoEDA_h
#define _eoEDA_h

//-----------------------------------------------------------------------------

#include <eoDistribution.h>

/** The abstract class for estimation of disribution algorithms.
 * This design evolve a probability distribution
 *  on the spaces of populations rather than a population
 *
 * It IS NOT an eoAlgo, as it evolves a distribution, not a population.
 *
 * @ingroup Algorithms
*/

template<class EOT> class eoEDA: public eoUF<eoDistribution<EOT>&, void>
{
};

#endif
