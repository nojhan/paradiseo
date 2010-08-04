#ifndef _doSamplerUniform_h
#define _doSamplerUniform_h

#include <utils/eoRNG.h>

#include "doSampler.h"
#include "doUniform.h"

/**
 * doSamplerUniform
 * This class uses the Uniform distribution parameters (bounds) to return
 * a random position used for population sampling.
 */
template < typename EOT >
class doSamplerUniform : public doSampler< doUniform< EOT > >
{
public:
    EOT sample( doUniform< EOT >& distrib )
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

#endif // !_doSamplerUniform_h
