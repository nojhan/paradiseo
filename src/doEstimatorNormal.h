#ifndef _doEstimatorNormal_h
#define _doEstimatorNormal_h

#include "doEstimator.h"
#include "doUniform.h"
#include "doStats.h"

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

	doCovMatrix cov(dimsize);

	for (unsigned int i = 0; i < popsize; ++i)
	    {
		cov.update(pop[i]);
	    }

	EOT mean(dimsize);
	EOT covariance(dimsize);

	for (unsigned int d = 0; d < dimsize; ++d)
	    {
		mean[d] = cov.get_mean(d);
		covariance[d] = cov.get_var(d);
	    }

	return doNormal< EOT >(mean, covariance);
    }
};

#endif // !_doEstimatorNormal_h
