// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doSamplerNormalMono_h
#define _doSamplerNormalMono_h

#include <utils/eoRNG.h>

#include "doSampler.h"
#include "doNormalMono.h"
#include "doBounder.h"

/**
 * doSamplerNormalMono
 * This class uses the NormalMono distribution parameters (bounds) to return
 * a random position used for population sampling.
 */
template < typename EOT >
class doSamplerNormalMono : public doSampler< doNormalMono< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    doSamplerNormalMono( doBounder< EOT > & bounder )
	: doSampler< doNormalMono< EOT > >( bounder )
    {}

    EOT sample( doNormalMono< EOT >& distrib )
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
		AtomType mean = distrib.mean()[i];
		AtomType variance = distrib.variance()[i];
	        AtomType random = rng.normal(mean, variance);

		assert(variance >= 0);

		solution.push_back(random);
	    }

	//-------------------------------------------------------------


	return solution;
    }
};

#endif // !_doSamplerNormalMono_h
