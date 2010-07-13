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

	//std::vector< doVar > var(dimsize);
	doCovMatrix cov(dimsize);

	for (unsigned int i = 0; i < popsize; ++i)
	    {
		cov.update(pop[i]);
		// for (unsigned int d = 0; d < dimsize; ++d)
		//     {
		// 	var[d].update(pop[i][d]);
		//     }
	    }

	EOT mean(dimsize);
	EOT variance(dimsize);

	for (unsigned int d = 0; d < dimsize; ++d)
	    {
		// mean[d] = var[d].get_mean();
		// variance[d] = var[d].get_var();
		//variance[d] = var[d].get_std(); // perhaps I should use this !?!

		mean[d] = cov.get_mean(d);
		variance[d] = cov.get_var(d);
	    }

	return doNormal< EOT >(mean, variance);
    }
};

#endif // !_doEstimatorNormal_h
