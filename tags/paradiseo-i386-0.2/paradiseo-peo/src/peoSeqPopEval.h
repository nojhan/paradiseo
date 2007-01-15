// "peoSeqPopEval.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __peoSeqPopEval_h
#define __peoSeqPopEval_h

#include <eoEvalFunc.h>

#include "peoPopEval.h"

//! Sequential evaluation functor wrapper.

//! The peoSeqPopEval class acts only as a ParadisEO specific sequential evaluation functor - a wrapper for incorporating
//! an <b>eoEvalFunc< EOT ></b>-derived class as evaluation functor. The specified EO evaluation object is applyied in an
//! iterative manner to each individual of a specified population.
template< class EOT > class peoSeqPopEval : public peoPopEval< EOT > {

public:

        //! Constructor function - it only sets an internal reference to point to the specified evaluation object.
	//!
	//! @param eoEvalFunc< EOT >& __eval - evaluation object to be applied for each individual of a specified population
	peoSeqPopEval( eoEvalFunc< EOT >& __eval );

	//! Operator for evaluating all the individuals of a given population - in a sequential iterative manner.
	//!
	//! @param eoPop< EOT >& __pop - population to be evaluated.
	void operator()( eoPop< EOT >& __pop );

private:

	eoEvalFunc< EOT >& eval;
};


template< class EOT > peoSeqPopEval< EOT > :: peoSeqPopEval( eoEvalFunc< EOT >& __eval ) : eval( __eval ) {

}


template< class EOT > void peoSeqPopEval< EOT > :: operator()( eoPop< EOT >& __pop ) {

	for ( unsigned i = 0; i < __pop.size(); i++ )
		eval( __pop[i] );
}


#endif
