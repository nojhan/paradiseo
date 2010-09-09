// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doEstimatorNormalMulti_h
#define _doEstimatorNormalMulti_h

#include "doEstimator.h"
#include "doNormalMulti.h"

template < typename EOT >
class doEstimatorNormalMulti : public doEstimator< doNormalMulti< EOT > >
{
public:
    class CovMatrix
    {
    public:
	typedef typename EOT::AtomType AtomType;

	CovMatrix( const eoPop< EOT >& pop )
	{
	    unsigned int p_size = pop.size(); // population size

	    assert(p_size > 0);

	    unsigned int s_size = pop[0].size(); // solution size

	    assert(s_size > 0);

	    ublas::matrix< AtomType > sample( p_size, s_size );

	    for (unsigned int i = 0; i < p_size; ++i)
		{
		    for (unsigned int j = 0; j < s_size; ++j)
			{
			    sample(i, j) = pop[i][j];
			}
		}

	    _varcovar.resize(s_size, s_size);


	    //-------------------------------------------------------------
	    // variance-covariance matrix are symmetric (and semi-definite
	    // positive), thus a triangular storage is sufficient
	    //
	    // variance-covariance matrix computation : transpose(A) * A
	    //-------------------------------------------------------------

	    ublas::symmetric_matrix< AtomType, ublas::lower > var = ublas::prod( ublas::trans( sample ), sample );

	    assert(var.size1() == s_size);
	    assert(var.size2() == s_size);
	    assert(var.size1() == _varcovar.size1());
	    assert(var.size2() == _varcovar.size2());

	    //-------------------------------------------------------------


	    // for (unsigned int i = 0; i < s_size; ++i)
	    // 	{
	    // 	    // triangular LOWER matrix, thus j is not going further than i
	    // 	    for (unsigned int j = 0; j <= i; ++j)
	    // 		{
	    // 		    // we want a reducted covariance matrix
	    // 		    _varcovar(i, j) = var(i, j) / p_size;
	    // 		}
	    // 	}

	    _varcovar = var / p_size;

	    _mean.resize(s_size);

	    // unit vector
	    ublas::scalar_vector< AtomType > u( p_size, 1 );

	    // sum over columns
	    _mean = ublas::prod( ublas::trans( sample ), u );

	    // division by n
	    _mean /= p_size;
	}

	const ublas::symmetric_matrix< AtomType, ublas::lower >& get_varcovar() const {return _varcovar;}

	const ublas::vector< AtomType >& get_mean() const {return _mean;}

    private:
	ublas::symmetric_matrix< AtomType, ublas::lower > _varcovar;
	ublas::vector< AtomType > _mean;
    };

public:
    typedef typename EOT::AtomType AtomType;

    doNormalMulti< EOT > operator()(eoPop<EOT>& pop)
    {
	unsigned int popsize = pop.size();
	assert(popsize > 0);

	unsigned int dimsize = pop[0].size();
	assert(dimsize > 0);

	CovMatrix cov( pop );

	return doNormalMulti< EOT >( cov.get_mean(), cov.get_varcovar() );
    }
};

#endif // !_doEstimatorNormalMulti_h
