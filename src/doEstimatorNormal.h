#ifndef _doEstimatorNormal_h
#define _doEstimatorNormal_h

#include "doEstimator.h"
#include "doUniform.h"
#include "doStats.h"

template < typename EOT >
class doEstimatorNormal : public doEstimator< doNormal< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    doNormal< EOT > operator()(eoPop<EOT>& pop)
    {
	unsigned int popsize = pop.size();
	assert(popsize > 0);

	unsigned int dimsize = pop[0].size();
	assert(dimsize > 0);

	//doCovMatrix cov(dimsize);
	doUblasCovMatrix< EOT > cov;

	cov.update(pop);

	return doNormal< EOT >(cov.get_mean(), cov.get_varcovar());
    }
};

#endif // !_doEstimatorNormal_h
