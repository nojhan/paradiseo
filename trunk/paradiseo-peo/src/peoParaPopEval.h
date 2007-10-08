// "peoParaPopEval.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __peoParaPopEval_h
#define __peoParaPopEval_h

#include <queue>
#include <eoEvalFunc.h>

#include "core/messaging.h"
#include "core/peo_debug.h"
#include "peoAggEvalFunc.h"
#include "peoNoAggEvalFunc.h"


//! Parallel evaluation functor wrapper.

//! The peoParaPopEval represents a wrapper for creating a functor capable of applying in parallel
//! an EO-derived evaluation functor. The class offers the possibility of chosing between a single-function evaluation
//! and an aggregate evaluation function, including several sub-evalution functions.
template< class EOT > class peoParaPopEval : public peoPopEval< EOT > {

public:

	using peoPopEval< EOT > :: requestResourceRequest;
	using peoPopEval< EOT > :: resume;
	using peoPopEval< EOT > :: stop;
	using peoPopEval< EOT > :: getOwner;
	
	//! Constructor function - an EO-derived evaluation functor has to be specified; an internal reference
	//! is set towards the specified evaluation functor.
	//!
	//! @param eoEvalFunc< EOT >& __eval_func - EO-derived evaluation functor to be applied in parallel on each individual of a specified population
	peoParaPopEval( eoEvalFunc< EOT >& __eval_func );

	//! Constructor function - a vector of EO-derived evaluation functors has to be specified as well as an aggregation function.
	//!
	//! @param const std :: vector< eoEvalFunc < EOT >* >& __funcs - vector of EO-derived partial evaluation functors;
	//! @param peoAggEvalFunc< EOT >& __merge_eval - aggregation functor for creating a fitness value out of the partial fitness values.
	peoParaPopEval( const std :: vector< eoEvalFunc < EOT >* >& __funcs, peoAggEvalFunc< EOT >& __merge_eval );

	//! Operator for applying the evaluation functor (direct or aggregate) for each individual of the specified population.	
	//!
	//! @param eoPop< EOT >& __pop - population to be evaluated by applying the evaluation functor specified in the constructor.
	void operator()( eoPop< EOT >& __pop );

	//! Auxiliary function for transferring data between the process requesting an evaluation operation and the process that
	//! performs the actual evaluation phase. There is no need to explicitly call the function.
	void packData();
	
	//! Auxiliary function for transferring data between the process requesting an evaluation operation and the process that
	//! performs the actual evaluation phase. There is no need to explicitly call the function.
	void unpackData();

	//! Auxiliary function - it calls the specified evaluation functor(s). There is no need to explicitly call the function.	
	void execute();
	
	//! Auxiliary function for transferring data between the process requesting an evaluation operation and the process that
	//! performs the actual evaluation phase. There is no need to explicitly call the function.
	void packResult();
	
	//! Auxiliary function for transferring data between the process requesting an evaluation operation and the process that
	//! performs the actual evaluation phase. There is no need to explicitly call the function.
	void unpackResult();
	
	//! Auxiliary function for notifications between the process requesting an evaluation operation and the processes that
	//! performs the actual evaluation phase. There is no need to explicitly call the function.
	void notifySendingData();

	//! Auxiliary function for notifications between the process requesting an evaluation operation and the processes that
	//! performs the actual evaluation phase. There is no need to explicitly call the function.
	void notifySendingAllResourceRequests();

private:


	const std :: vector< eoEvalFunc < EOT >* >& funcs;
	std :: vector< eoEvalFunc < EOT >* > one_func;
	
	peoAggEvalFunc< EOT >& merge_eval;
	peoNoAggEvalFunc< EOT > no_merge_eval;
	
	std :: queue< EOT* >tasks;
	
	std :: map< EOT*, std :: pair< unsigned, unsigned > > progression;
	
	unsigned num_func;
	
	EOT sol;
	
	EOT *ad_sol;
	
	unsigned total;
};


template< class EOT > peoParaPopEval< EOT > :: peoParaPopEval( eoEvalFunc< EOT >& __eval_func ) : 

		funcs( one_func ), merge_eval( no_merge_eval )
{

	one_func.push_back( &__eval_func );
}


template< class EOT > peoParaPopEval< EOT > :: peoParaPopEval( 

				const std :: vector< eoEvalFunc< EOT >* >& __funcs,
				peoAggEvalFunc< EOT >& __merge_eval 

		) : funcs( __funcs ), merge_eval( __merge_eval )
{

}


template< class EOT > void peoParaPopEval< EOT >::operator()( eoPop< EOT >& __pop ) {
  for ( unsigned i = 0; i < __pop.size(); i++ ) {
    __pop[ i ].fitness(typename EOT :: Fitness() );  	
		progression[ &__pop[ i ] ].first = funcs.size() - 1;
		progression[ &__pop[ i ] ].second = funcs.size();
		for ( unsigned j = 0; j < funcs.size(); j++ ) {
			/* Queuing the 'invalid' solution and its associated owner */
			tasks.push( &__pop[ i ] );
		}
	}
	total = funcs.size() * __pop.size();
	requestResourceRequest( funcs.size() * __pop.size() );
	stop();
}


template< class EOT > void peoParaPopEval< EOT > :: packData() {
	//  printDebugMessage ("debut pakc data");
	pack( progression[ tasks.front() ].first-- );
	
	/* Packing the contents :-) of the solution */
	pack( *tasks.front() );
	
	/* Packing the addresses of both the solution and the owner */
	pack( tasks.front() );
	tasks.pop(  );
}


template< class EOT > void peoParaPopEval< EOT > :: unpackData() {
	unpack( num_func );
	/* Unpacking the solution */
	unpack( sol );
	/* Unpacking the @ of that one */
	unpack( ad_sol );
}


template< class EOT > void peoParaPopEval< EOT > :: execute() {
	/* Computing the fitness of the solution */
  funcs[ num_func ]->operator()( sol );
}


template< class EOT > void peoParaPopEval< EOT > :: packResult() {
  /* Packing the fitness of the solution */
	pack( sol.fitness() );
	/* Packing the @ of the individual */
	pack( ad_sol );
}


template< class EOT > void peoParaPopEval< EOT > :: unpackResult() {
	typename EOT :: Fitness fit;
	
	/* Unpacking the computed fitness */
	unpack( fit );
		
	/* Unpacking the @ of the associated individual */
	unpack( ad_sol );
	
	
	/* Associating the fitness the local solution */
	merge_eval( *ad_sol, fit );

	progression[ ad_sol ].second--;

	/* Notifying the container of the termination of the evaluation */
	if ( !progression[ ad_sol ].second ) {

		progression.erase( ad_sol );
	}
	
	total--;
	if ( !total ) {

		getOwner()->setActive();
		resume();
	}
}


template< class EOT > void peoParaPopEval< EOT > :: notifySendingData() {
}


template< class EOT > void peoParaPopEval< EOT > :: notifySendingAllResourceRequests() {
	getOwner()->setPassive();
}


#endif
