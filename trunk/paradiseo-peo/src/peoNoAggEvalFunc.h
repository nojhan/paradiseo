// "peoNoAggEvalFunc.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __peoNoAggEvalFunc_h
#define __peoNoAggEvalFunc_h

#include "peoAggEvalFunc.h"

//! Class providing a simple interface for associating a fitness value to a specified individual.

//! The peoNoAggEvalFunc class does nothing more than an association between a fitness value and a specified individual.
//! The class is provided as a mean of declaring that no aggregation is required for the evaluation function - the fitness
//! value is explicitly specified.
template< class EOT > class peoNoAggEvalFunc : public peoAggEvalFunc< EOT > {

public :

	//! Operator which sets as fitness the <b>__fit</b> value for the <b>__sol</b> individual
	void operator()( EOT& __sol, const typename EOT :: Fitness& __fit );
};


template< class EOT > void peoNoAggEvalFunc< EOT > :: operator()( EOT& __sol, const typename EOT :: Fitness& __fit ) {

	__sol.fitness( __fit );
}


#endif
