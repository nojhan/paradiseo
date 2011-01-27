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

#ifndef _edoSamplerUniform_h
#define _edoSamplerUniform_h

#include <utils/eoRNG.h>

#include "edoSampler.h"
#include "edoUniform.h"

/**
 * edoSamplerUniform
 * This class uses the Uniform distribution parameters (bounds) to return
 * a random position used for population sampling.
 */
template < typename EOT, class D=edoUniform<EOT> > // FIXME: D template name is there really used ?!?
class edoSamplerUniform : public edoSampler< edoUniform< EOT > >
{
public:
    typedef D Distrib;

    edoSamplerUniform(edoBounder< EOT > & bounder)
	: edoSampler< edoUniform<EOT> >(bounder) // FIXME: Why D is not used here ?
    {}

    /*
    edoSamplerUniform()
	: edoSampler< edoUniform<EOT> >()
    {}
    */

    EOT sample( edoUniform< EOT >& distrib )
    {
	unsigned int size = distrib.size();
	assert(size > 0);


	//-------------------------------------------------------------
	// Point we want to sample to get higher a set of points
	// (coordinates in n dimension)
	// x = {x1, x2, ..., xn}
	//-------------------------------------------------------------

	EOT solution;

	//-------------------------------------------------------------


	//-------------------------------------------------------------
	// Sampling all dimensions
	//-------------------------------------------------------------

	for (unsigned int i = 0; i < size; ++i)
	    {
		double min = distrib.min()[i];
		double max = distrib.max()[i];
		double random = rng.uniform(min, max);

		assert(min <= random && random <= max);

		solution.push_back(random);
	    }

	//-------------------------------------------------------------


	return solution;
    }
};

#endif // !_edoSamplerUniform_h
