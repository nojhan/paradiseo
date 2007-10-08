/* 
* <main.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
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

#include "route.h"
#include "route_init.h"
#include "route_eval.h"

#include "order_xover.h"
#include "city_swap.h"

#include "param.h"


#include <peo>


#define POP_SIZE 10
#define NUM_GEN 100
#define CROSS_RATE 1.0
#define MUT_RATE 0.01

#define MIG_FREQ 10
#define MIG_SIZE 5


int main( int __argc, char** __argv ) {

	// initializing the ParadisEO-PEO environment
	peo :: init( __argc, __argv );


	// processing the command line specified parameters
	loadParameters( __argc, __argv );


	// init, eval operators, EA operators -------------------------------------------------------------------------------------------------------------

	RouteInit route_init;	// random init object - creates random Route objects
	RouteEval full_eval;	// evaluator object - offers a fitness value for a specified Route object

	OrderXover crossover;	// crossover operator - creates two offsprings out of two specified parents
	CitySwap mutation;	// mutation operator - randomly mutates one gene for a specified individual
	// ------------------------------------------------------------------------------------------------------------------------------------------------


	// evolutionary algorithm components --------------------------------------------------------------------------------------------------------------

	eoPop< Route > population( POP_SIZE, route_init );	// initial population for the algorithm having POP_SIZE individuals
	peoParaPopEval< Route > eaPopEval( full_eval );		// evaluator object - to be applied at each iteration on the entire population

	eoGenContinue< Route > eaCont( NUM_GEN );		// continuation criterion - the algorithm will iterate for NUM_GEN generations
	eoCheckPoint< Route > eaCheckpointContinue( eaCont );	// checkpoint object - verify at each iteration if the continuation criterion is met

	eoRankingSelect< Route > selectionStrategy;		// selection strategy - applied at each iteration for selecting parent individuals
	eoSelectNumber< Route > eaSelect( selectionStrategy, POP_SIZE ); // selection object - POP_SIZE individuals are selected at each iteration

	// transform operator - includes the crossover and the mutation operators with a specified associated rate
	eoSGATransform< Route > transform( crossover, CROSS_RATE, mutation, MUT_RATE );
	peoSeqTransform< Route > eaTransform( transform );	// ParadisEO transform operator (please remark the peo prefix) - wraps an e EO transform object

	eoPlusReplacement< Route > eaReplace;			// replacement strategy - for replacing the initial population with offspring individuals
	// ------------------------------------------------------------------------------------------------------------------------------------------------



	RingTopology topology;

	// migration policy and components ----------------------------------------------------------------------------------------------------------------

	eoPeriodicContinue< Route > mig_cont( MIG_FREQ ); 	// migration occurs periodically

	eoRandomSelect< Route > mig_select_one; 		// emigrants are randomly selected 
	eoSelectNumber< Route > mig_select( mig_select_one, MIG_SIZE );

	eoPlusReplacement< Route > mig_replace; 		// immigrants replace the worse individuals

	peoSyncIslandMig< Route > mig( MIG_FREQ, mig_select, mig_replace, topology, population, population );
	//peoAsyncIslandMig< Route > mig( mig_cont, mig_select, mig_replace, topology, population, population );

	eaCheckpointContinue.add( mig );
	// ------------------------------------------------------------------------------------------------------------------------------------------------





	// ParadisEO-PEO evolutionary algorithm -----------------------------------------------------------------------------------------------------------

	peoEA< Route > eaAlg( eaCheckpointContinue, eaPopEval, eaSelect, eaTransform, eaReplace );

	mig.setOwner( eaAlg );
	
	eaAlg( population );	// specifying the initial population for the algorithm, to be iteratively evolved
	// ------------------------------------------------------------------------------------------------------------------------------------------------




	// evolutionary algorithm components --------------------------------------------------------------------------------------------------------------

	eoPop< Route > population2( POP_SIZE, route_init );	// initial population for the algorithm having POP_SIZE individuals
	peoParaPopEval< Route > eaPopEval2( full_eval );	// evaluator object - to be applied at each iteration on the entire population

	eoGenContinue< Route > eaCont2( NUM_GEN );		// continuation criterion - the algorithm will iterate for NUM_GEN generations
	eoCheckPoint< Route > eaCheckpointContinue2( eaCont2 );	// checkpoint object - verify at each iteration if the continuation criterion is met

	eoRankingSelect< Route > selectionStrategy2;		// selection strategy - applied at each iteration for selecting parent individuals
	eoSelectNumber< Route > eaSelect2( selectionStrategy2, POP_SIZE ); // selection object - POP_SIZE individuals are selected at each iteration

	// transform operator - includes the crossover and the mutation operators with a specified associated rate
	eoSGATransform< Route > transform2( crossover, CROSS_RATE, mutation, MUT_RATE );
	peoSeqTransform< Route > eaTransform2( transform2 );	// ParadisEO transform operator (please remark the peo prefix) - wraps an e EO transform object

	eoPlusReplacement< Route > eaReplace2;			// replacement strategy - for replacing the initial population with offspring individuals
	// ------------------------------------------------------------------------------------------------------------------------------------------------




	// migration policy and components ----------------------------------------------------------------------------------------------------------------

	eoPeriodicContinue< Route > mig_cont2( MIG_FREQ ); 	// migration occurs periodically

	eoRandomSelect< Route > mig_select_one2; 		// emigrants are randomly selected 
	eoSelectNumber< Route > mig_select2( mig_select_one2, MIG_SIZE );

	eoPlusReplacement< Route > mig_replace2; 		// immigrants replace the worse individuals

	peoSyncIslandMig< Route > mig2( MIG_FREQ, mig_select2, mig_replace2, topology, population2, population2 );
	//peoAsyncIslandMig< Route > mig2( mig_cont2, mig_select2, mig_replace2, topology, population2, population2 );

	eaCheckpointContinue2.add( mig2 );
	// ------------------------------------------------------------------------------------------------------------------------------------------------





	// ParadisEO-PEO evolutionary algorithm -----------------------------------------------------------------------------------------------------------

	peoEA< Route > eaAlg2( eaCheckpointContinue2, eaPopEval2, eaSelect2, eaTransform2, eaReplace2 );

	mig2.setOwner( eaAlg2 );
	
	eaAlg2( population2 );	// specifying the initial population for the algorithm, to be iteratively evolved
	// ------------------------------------------------------------------------------------------------------------------------------------------------



	peo :: run( );
	peo :: finalize( );
	// shutting down the ParadisEO-PEO environment

	return 0;
}
