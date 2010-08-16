#ifndef _doEstimatorNormalMono_h
#define _doEstimatorNormalMono_h

#include "doEstimator.h"
#include "doNormalMono.h"

template < typename EOT >
class doEstimatorNormalMono : public doEstimator< doNormalMono< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    class Variance
    {
    public:
	Variance() : _sumvar(0){}

	void update(AtomType v)
	{
	    _n++;

	    AtomType d = v - _mean;

	    _mean += 1 / _n * d;
	    _sumvar += (_n - 1) / _n * d * d;
	}

	AtomType get_mean() const {return _mean;}
	AtomType get_var() const {return _sumvar / (_n - 1);}
	AtomType get_std() const {return sqrt( get_var() );}

    private:
	AtomType _n;
	AtomType _mean;
	AtomType _sumvar;
    };

public:
    doNormalMono< EOT > operator()(eoPop<EOT>& pop)
    {
	unsigned int popsize = pop.size();
	assert(popsize > 0);

	unsigned int dimsize = pop[0].size();
	assert(dimsize > 0);

	std::vector< Variance > var( dimsize );

	for (unsigned int i = 0; i < popsize; ++i)
	    {
		for (unsigned int d = 0; d < dimsize; ++d)
		    {
			var[d].update( pop[i][d] );
		    }
	    }

	EOT mean( dimsize );
	EOT variance( dimsize );

	for (unsigned int d = 0; d < dimsize; ++d)
	    {
		mean[d] = var[d].get_mean();
		variance[d] = var[d].get_var();
	    }

	return doNormalMono< EOT >( mean, variance );
    }
};

#endif // !_doEstimatorNormalMono_h
