#ifndef _doSamplerNormal_h
#define _doSamplerNormal_h

#include <utils/eoRNG.h>

#include "doSampler.h"
#include "doNormal.h"
#include "doBounder.h"
#include "doStats.h"

/**
 * doSamplerNormal
 * This class uses the Normal distribution parameters (bounds) to return
 * a random position used for population sampling.
 */
template < typename EOT >
class doSamplerNormal : public doSampler< doNormal< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    doSamplerNormal( doBounder< EOT > & bounder )
	: doSampler< doNormal< EOT > >( bounder )
    {}

    EOT sample( doNormal< EOT >& distrib )
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
		Cholesky< EOT > cholesky;

		cholesky.update( distrib.varcovar() );

		// solution.push_back(
		// 		   rng.normal(distrib.mean()[i],
		// 			      distrib.varcovar()[i])
		// 		   );

		//rng.normal() + 
	    }

	//-------------------------------------------------------------

	return solution;
    }
};

#endif // !_doSamplerNormal_h
