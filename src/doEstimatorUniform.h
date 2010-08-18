// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doEstimatorUniform_h
#define _doEstimatorUniform_h

#include "doEstimator.h"
#include "doUniform.h"

// TODO: calcule de la moyenne + covariance dans une classe derivee

template < typename EOT >
class doEstimatorUniform : public doEstimator< doUniform< EOT > >
{
public:
    doUniform< EOT > operator()(eoPop<EOT>& pop)
    {
	unsigned int size = pop.size();

	assert(size > 0);

	EOT min = pop[0];
	EOT max = pop[0];

	for (unsigned int i = 1; i < size; ++i)
	    {
		unsigned int size = pop[i].size();

		assert(size > 0);

		// possibilité d'utiliser std::min_element et std::max_element mais exige 2 pass au lieu d'1.

		for (unsigned int d = 0; d < size; ++d)
		    {
			if (pop[i][d] < min[d])
			    min[d] = pop[i][d];

			if (pop[i][d] > max[d])
			    max[d] = pop[i][d];
		    }
	    }

	return doUniform< EOT >(min, max);
    }
};

#endif // !_doEstimatorUniform_h
