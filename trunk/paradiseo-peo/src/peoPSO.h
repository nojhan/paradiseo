/* <peoPSO.h>
*
*  (c) OPAC Team, October 2007
*
* Clive Canape
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
*   Contact: clive.canape@inria.fr
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

//! Class providing an elementary ParadisEO evolutionary algorithm.

//! The peoPSO class offers an elementary Particle Swarm Optimization implementation. In addition, as compared
//! with the algorithms provided by the EO framework, the peoPSO class has the underlying necessary structure
//! for including, for example, parallel evaluation, etc. 

template < class POT > class peoPSO : public Runner {

public:


  //! Constructor for the Particle Swarm Optimization
  //! @param eoContinue< POT >& __cont - continuation criterion specifying whether the algorithm should continue or not;
  //! @param peoPopEval< POT >& __pop_eval - evaluation operator; it allows the specification of parallel evaluation operators, aggregate evaluation functions, etc.;
  //! @param eoVelocity< POT >& __velocity - velocity operator;
  //! @param eoFlight< POT >& __flight - flight operator;
  peoPSO(
	 eoContinue< POT >& __cont,
	 peoPopEval< POT >& __pop_eval,
	 eoVelocity < POT > &_velocity,
	 eoFlight < POT > &_flight);

  //! Particle Swarm Optimization function - a side effect of the fact that the class is derived from the Runner class
  //! thus requiring the existence of a run function, the algorithm being executed on a distinct thread.
	void run();
	
  //! Function operator for specifying the population to be associated with the algorithm.
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
