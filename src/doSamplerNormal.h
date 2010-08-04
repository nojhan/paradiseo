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
	// Cholesky factorisation gererating matrix L from covariance
	// matrix V.
	// We must use cholesky.get_L() to get the resulting matrix.
	//
	// L = cholesky decomposition of varcovar
	//-------------------------------------------------------------

	Cholesky< EOT > cholesky;
	cholesky.update( distrib.varcovar() );
	ublas::symmetric_matrix< AtomType, ublas::lower > L = cholesky.get_L();

	//-------------------------------------------------------------


	//-------------------------------------------------------------
	// T = vector of size elements drawn in N(0,1) rng.normal(1.0)
	//-------------------------------------------------------------

	ublas::vector< AtomType > T( size );

	for ( unsigned int i = 0; i < size; ++i )
	    {
		T( i ) = rng.normal( 1.0 );
	    }

	//-------------------------------------------------------------


	//-------------------------------------------------------------
	// LT = prod( L, trans(T) ) ?
	// LT = prod( L, T )
	//-------------------------------------------------------------

	//ublas::symmetric_matrix< AtomType, ublas::lower > LT = ublas::prod( L, ublas::trans( T ) );
	ublas::vector< AtomType > LT = ublas::prod( L, T );

	//-------------------------------------------------------------


	//-------------------------------------------------------------
	// solution = means + trans( LT ) ?
	// solution = means + LT
	//-------------------------------------------------------------

	ublas::vector< AtomType > mean = distrib.mean();

	ublas::vector< AtomType > ublas_solution = mean + LT;
	//ublas::vector< AtomType > ublas_solution = mean + ublas::trans( LT );

	EOT solution( size );

	std::copy( ublas_solution.begin(), ublas_solution.end(), solution.begin() );

	//-------------------------------------------------------------

	return solution;
    }
};

#endif // !_doSamplerNormal_h
