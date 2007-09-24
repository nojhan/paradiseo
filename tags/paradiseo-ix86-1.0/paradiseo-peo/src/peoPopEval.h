// "peoPopEval.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __peoPopEval_h
#define __peoPopEval_h

#include "core/service.h"

//! Interface for ParadisEO specific evaluation functors.

//! The <b>peoPopEval</b> class provides the interface for constructing ParadisEO specific evaluation functors.
//! The derived classes may be used as wrappers for <b>EO</b>-derived evaluation functors. In order to have an example,
//! please refer to the implementation of the <b>peoSeqPopEval</b> and <b>peoParaPopEval</b> classes.
template< class EOT > class peoPopEval : public Service {

public:

	//! Interface function providing the signature for constructing an evaluation functor.
	virtual void operator()( eoPop< EOT >& __pop ) = 0;
};


#endif
