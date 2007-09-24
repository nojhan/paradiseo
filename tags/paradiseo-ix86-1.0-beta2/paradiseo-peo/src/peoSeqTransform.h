// "peoSeqTransform.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __peoSeqTransform_h
#define __peoSeqTransform_h

#include "peoTransform.h"


//! ParadisEO specific wrapper class offering the possibility of using EO derived transform operators.

//! The peoSeqTransform represent a wrapper for offering the possibility of using EO derived transform operators
//! along with the ParadisEO evolutionary algorithms. A minimal set of interface functions is also provided for creating the
//! link with the parallel architecture of the ParadisEO framework.
template< class EOT > class peoSeqTransform : public peoTransform< EOT > {

public:

	//! Constructor function - sets an internal reference towards the specified EO-derived transform object.
	//!
	//! @param eoTransform< EOT >& __trans - EO-derived transform object including crossover and mutation operators.
	peoSeqTransform( eoTransform< EOT >& __trans );
	
	//! Operator for applying the specified transform operators on each individual of the given population.
	//!
	//! @param eoPop< EOT >& __pop - population to be transformed by applying the crossover and mutation operators.
	void operator()( eoPop< EOT >& __pop );
	
	//! Interface function for providing a link with the parallel architecture of the ParadisEO framework.
	virtual void packData() { }

	//! Interface function for providing a link with the parallel architecture of the ParadisEO framework.
	virtual void unpackData() { }
	
	//! Interface function for providing a link with the parallel architecture of the ParadisEO framework.
	virtual void execute() { }
	
	//! Interface function for providing a link with the parallel architecture of the ParadisEO framework.
	virtual void packResult() { }

	//! Interface function for providing a link with the parallel architecture of the ParadisEO framework.
	virtual void unpackResult() { }

private:

	eoTransform< EOT >& trans;
};


template< class EOT > peoSeqTransform< EOT > :: peoSeqTransform( eoTransform< EOT >& __trans ) : trans( __trans ) {

}


template< class EOT > void peoSeqTransform< EOT > :: operator()( eoPop< EOT >& __pop ) {

	trans( __pop );
}


#endif
