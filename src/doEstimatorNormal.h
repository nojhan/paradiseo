#ifndef _doEstimatorNormal_h
#define _doEstimatorNormal_h

#include "doEstimator.h"
#include "doUniform.h"
#include "doStats.h"

// TODO: calcule de la moyenne + covariance dans une classe derivee

template < typename EOT >
class doEstimatorNormal : public doEstimator< doNormal< EOT > >
{
public:
    doNormal< EOT > operator()(eoPop<EOT>& pop)
    {
	unsigned int popsize = pop.size();
	assert(popsize > 0);

	unsigned int dimsize = pop[0].size();
	assert(dimsize > 0);

	std::vector< Var > var(dimsize);

	for (unsigned int i = 0; i < popsize; ++i)
	    {
		for (unsigned int d = 0; d < dimsize; ++d)
		    {
			var[d].update(pop[i][d]);
		    }
	    }

	EOT mean(dimsize);
	EOT variance(dimsize);

	for (unsigned int d = 0; d < dimsize; ++d)
	    {
		mean[d] = var[d].get_mean();
		variance[d] = var[d].get_var();
		//variance[d] = var[d].get_std(); // perhaps I should use this !?!
	    }

	return doNormal< EOT >(mean, variance);
    }
};

#endif // !_doEstimatorNormal_h
