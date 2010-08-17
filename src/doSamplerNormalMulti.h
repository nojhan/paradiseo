#ifndef _doSamplerNormalMulti_h
#define _doSamplerNormalMulti_h

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <utils/eoRNG.h>

#include "doSampler.h"
#include "doNormalMulti.h"
#include "doBounder.h"

/**
 * doSamplerNormalMulti
 * This class uses the Normal distribution parameters (bounds) to return
 * a random position used for population sampling.
 */
template < typename EOT >
class doSamplerNormalMulti : public doSampler< doNormalMulti< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    class Cholesky
    {
    public:
	Cholesky( const ublas::symmetric_matrix< AtomType, ublas::lower >& V)
	{
	    unsigned int Vl = V.size1();

	    assert(Vl > 0);

	    unsigned int Vc = V.size2();

	    assert(Vc > 0);

	    assert( Vl == Vc );

	    _L.resize(Vl, Vc);

	    unsigned int i,j,k;

	    // first column
	    i=0;

	    // diagonal
	    j=0;
	    _L(0, 0) = sqrt( V(0, 0) );

	    // end of the column
	    for ( j = 1; j < Vc; ++j )
		{
		    _L(j, 0) = V(0, j) / _L(0, 0);
		}

	    // end of the matrix
	    for ( i = 1; i < Vl; ++i ) // each column
		{

		    // diagonal
		    double sum = 0.0;

		    for ( k = 0; k < i; ++k)
			{
			    sum += _L(i, k) * _L(i, k);
			}

		    // assert( ( V(i, i) - sum ) > 0 );

		    //_L(i, i) = sqrt( V(i, i) - sum );

		    _L(i,i) = sqrt( fabs( V(i,i) - sum) );

		    for ( j = i + 1; j < Vl; ++j ) // rows
			{
			    // one element
			    sum = 0.0;

			    for ( k = 0; k < i; ++k )
				{
				    sum += _L(j, k) * _L(i, k);
				}

			    _L(j, i) = (V(j, i) - sum) / _L(i, i);
			}
		}
	}

	const ublas::symmetric_matrix< AtomType, ublas::lower >& get_L() const {return _L;}

    private:
	ublas::symmetric_matrix< AtomType, ublas::lower > _L;
    };

    doSamplerNormalMulti( doBounder< EOT > & bounder )
	: doSampler< doNormalMulti< EOT > >( bounder )
    {}

    EOT sample( doNormalMulti< EOT >& distrib )
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

	Cholesky cholesky( distrib.varcovar() );
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
	// LT = prod( L, T )
	//-------------------------------------------------------------

	ublas::vector< AtomType > LT = ublas::prod( L, T );

	//-------------------------------------------------------------


	//-------------------------------------------------------------
	// solution = means + LT
	//-------------------------------------------------------------

	ublas::vector< AtomType > mean = distrib.mean();

	ublas::vector< AtomType > ublas_solution = mean + LT;

	EOT solution( size );

	std::copy( ublas_solution.begin(), ublas_solution.end(), solution.begin() );

	//-------------------------------------------------------------

	return solution;
    }
};

#endif // !_doSamplerNormalMulti_h
