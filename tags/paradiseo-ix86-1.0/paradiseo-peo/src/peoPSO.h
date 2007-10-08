// "peoPSO.h"

// (c) OPAC Team, October 2007

/* 
   Contact: clive.canape@inria.fr
*/

#ifndef __peoPSO_h
#define __peoPSO_h

#include <eoContinue.h> 
#include <eoEvalFunc.h> 
#include <eoPopEvalFunc.h>
#include <eoPSO.h>
#include <eoVelocity.h>
#include <eoFlight.h>
#include "peoPopEval.h"
#include "core/runner.h" 
#include "core/peo_debug.h" 


template < class POT > class peoPSO : public Runner {

public:


  // Constructor for the Particle Swarm Optimization
  peoPSO(
	 eoContinue< POT >& __cont,
	 peoPopEval< POT >& __pop_eval,
	 eoVelocity < POT > &_velocity,
	 eoFlight < POT > &_flight);

  // Particle Swarm Optimization function - a side effect of the fact that the class is derived from the Runner class
  // thus requiring the existence of a run function, the algorithm being executed on a distinct thread.
	void run();
	
  // Function operator for specifying the population to be associated with the algorithm.
	void operator()( eoPop< POT >& __pop );

private:

	eoContinue< POT >& cont;
	peoPopEval< POT >& pop_eval;
	eoPop< POT >* pop;
	eoVelocity < POT > &velocity;
	eoFlight < POT > &flight;
};


template < class POT > peoPSO< POT > :: peoPSO( 

				eoContinue< POT >& __cont, 
				peoPopEval< POT >& __pop_eval, 
				eoVelocity < POT > &__velocity,
				eoFlight < POT > &__flight
				) : cont( __cont ), pop_eval(__pop_eval ),velocity( __velocity),flight( __flight)
{
	pop_eval.setOwner( *this );
}


template< class POT > void peoPSO< POT > :: operator ()( eoPop< POT >& __pop ) {

	pop = &__pop;
}


template< class POT > void peoPSO< POT > :: run() {

	printDebugMessage( "Performing the first evaluation of the population." );
	do {	
     	        printDebugMessage( "Performing the velocity evaluation." );
		velocity.apply ( *pop );
		printDebugMessage( "Performing the flight." );
		flight.apply ( *pop );
		printDebugMessage( "Performing the evaluation." );
		pop_eval(*pop);
		velocity.updateNeighborhood( *pop );
	} while ( cont( *pop ) );
}


#endif
