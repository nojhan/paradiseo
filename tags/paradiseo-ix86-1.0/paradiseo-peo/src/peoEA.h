/* 
* <peoEA.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Alexandru-Adrian Tantar
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#ifndef __peoEA_h
#define __peoEA_h

#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoSelect.h>
#include <eoPopEvalFunc.h>
#include <eoReplacement.h>

#include "peoPopEval.h"
#include "peoTransform.h"
#include "core/runner.h"
#include "core/peo_debug.h"

//! Class providing an elementary ParadisEO evolutionary algorithm.

//! The peoEA class offers an elementary evolutionary algorithm implementation. In addition, as compared
//! with the algorithms provided by the EO framework, the peoEA class has the underlying necessary structure
//! for including, for example, parallel evaluation and parallel transformation operators, migration operators
//! etc. Although there is no restriction on using the algorithms provided by the EO framework, the drawback resides 
//! in the fact that the EO implementation is exclusively sequential and, in consequence, no parallelism is provided.
//! A simple example for constructing a peoEA object:
//!
//!	<table style="border:none; border-spacing:0px;text-align:left; vertical-align:top; font-size:8pt;" border="0">
//!	<tr><td>... &nbsp;</td> <td> &nbsp; </td></tr>
//!	<tr><td>eoPop< EOT > population( POP_SIZE, popInitializer ); &nbsp;</td> <td>// creation of a population with POP_SIZE individuals - the popInitializer is a functor to be called for each individual</td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>eoGenContinue< EOT > eaCont( NUM_GEN ); &nbsp;</td> <td>// number of generations for the evolutionary algorithm</td></tr>
//!	<tr><td>eoCheckPoint< EOT > eaCheckpointContinue( eaCont ); &nbsp;</td> <td>// checkpoint incorporating the continuation criterion - startpoint for adding other checkpoint objects</td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>peoSeqPopEval< EOT > eaPopEval( evalFunction ); &nbsp;</td> <td>// sequential evaluation functor wrapper - evalFunction represents the actual evaluation functor </td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>eoRankingSelect< EOT > selectionStrategy; &nbsp;</td> <td>// selection strategy for creating the offspring population - a simple ranking selection in this case </td></tr>
//!	<tr><td>eoSelectNumber< EOT > eaSelect( selectionStrategy, POP_SIZE ); &nbsp;</td> <td>// the number of individuals to be selected for creating the offspring population </td></tr>
//!	<tr><td>eoRankingSelect< EOT > selectionStrategy; &nbsp;</td> <td>// selection strategy for creating the offspring population - a simple ranking selection in this case </td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>eoSGATransform< EOT > transform( crossover, CROSS_RATE, mutation, MUT_RATE ); &nbsp;</td> <td>// transformation operator - crossover and mutation operators with their associated probabilities </td></tr>
//!	<tr><td>peoSeqTransform< EOT > eaTransform( transform ); &nbsp;</td> <td>// ParadisEO specific sequential operator - a parallel version may be specified in the same manner </td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>eoPlusReplacement< EOT > eaReplace; &nbsp;</td> <td>// replacement strategy - for integrating the offspring resulting individuals in the initial population </td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>peoEA< EOT > eaAlg( eaCheckpointContinue, eaPopEval, eaSelect, eaTransform, eaReplace ); &nbsp;</td> <td>// ParadisEO evolutionary algorithm integrating the above defined objects </td></tr>
//!	<tr><td>eaAlg( population ); &nbsp;</td> <td>// specifying the initial population for the algorithm </td></tr>
//!	<tr><td>... &nbsp;</td> <td> &nbsp; </td></tr>
//!	</table>
template < class EOT > class peoEA : public Runner {

public:

	//! Constructor for the evolutionary algorithm object - several basic parameters have to be specified,
	//! allowing for different levels of parallelism. Depending on the requirements, a sequential or a parallel
	//! evaluation operator may be specified or, in the same manner, a sequential or a parallel transformation
	//! operator may be given as parameter. Out of the box objects may be provided, from the EO package, for example, 
	//! or custom defined ones may be specified, provided that they are derived from the correct base classes.
	//!
	//! @param eoContinue< EOT >& __cont - continuation criterion specifying whether the algorithm should continue or not;
	//! @param peoPopEval< EOT >& __pop_eval - evaluation operator; it allows the specification of parallel evaluation operators, aggregate evaluation functions, etc.;
	//! @param eoSelect< EOT >& __select - selection strategy to be applied for constructing a list of offspring individuals;
	//! @param peoTransform< EOT >& __trans - transformation operator, i.e. crossover and mutation; allows for sequential or parallel transform;
	//! @param eoReplacement< EOT >& __replace - replacement strategy for integrating the offspring individuals in the initial population;
	peoEA( 
		eoContinue< EOT >& __cont,
		peoPopEval< EOT >& __pop_eval,
		eoSelect< EOT >& __select,
		peoTransform< EOT >& __trans,
		eoReplacement< EOT >& __replace 
	);

	//! Evolutionary algorithm function - a side effect of the fact that the class is derived from the <b>Runner</b> class,
	//! thus requiring the existence of a <i>run</i> function, the algorithm being executed on a distinct thread.
	void run();
	
	//! Function operator for specifying the population to be associated with the algorithm.
	//!
	//! @param eoPop< EOT >& __pop - initial population of the algorithm, to be iteratively evolved;
	void operator()( eoPop< EOT >& __pop );

private:


	eoContinue< EOT >& cont;
	peoPopEval< EOT >& pop_eval;
	eoSelect< EOT >& select;
	peoTransform< EOT >& trans;
	eoReplacement< EOT >& replace;
	eoPop< EOT >* pop;
};


template < class EOT > peoEA< EOT > :: peoEA( 

				eoContinue< EOT >& __cont, 
				peoPopEval< EOT >& __pop_eval, 
				eoSelect< EOT >& __select, 
				peoTransform< EOT >& __trans, 
				eoReplacement< EOT >& __replace

		) : cont( __cont ), pop_eval( __pop_eval ), select( __select ), trans( __trans ), replace( __replace )
{

	trans.setOwner( *this );
	pop_eval.setOwner( *this );
}


template< class EOT > void peoEA< EOT > :: operator ()( eoPop< EOT >& __pop ) {

	pop = &__pop;
}


template< class EOT > void peoEA< EOT > :: run() {

	printDebugMessage( "performing the first evaluation of the population." );
	pop_eval( *pop );
	
	do {

		eoPop< EOT > off;

		printDebugMessage( "performing the selection step." );
		select( *pop, off );
		trans( off );

		printDebugMessage( "performing the evaluation of the population." );
		pop_eval( off );

		printDebugMessage( "performing the replacement of the population." );
		replace( *pop, off );

		printDebugMessage( "deciding of the continuation." );
	
	} while ( cont( *pop ) );
}


#endif
