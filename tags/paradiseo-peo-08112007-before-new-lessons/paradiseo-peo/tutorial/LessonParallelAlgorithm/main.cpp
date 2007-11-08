// "main.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <eo>
#include <paradiseo>

#include <peoParallelAlgorithmWrapper.h>
#include <peoSynchronousMultiStart.h>



#include "route.h"
#include "route_init.h"
#include "route_eval.h"

#include "order_xover.h"
#include "city_swap.h"

#include "param.h"




#include <mo.h>

#include <graph.h>
#include <route.h>
#include <route_eval.h>
#include <route_init.h>

#include <two_opt.h>
#include <two_opt_init.h>
#include <two_opt_next.h>
#include <two_opt_incr_eval.h>



#define RANDOM_POP_SIZE	30
#define RANDOM_ITERATIONS 10


#define POP_SIZE 10
#define NUM_GEN 100
#define CROSS_RATE 1.0
#define MUT_RATE 0.01

#define NUMBER_OF_POPULATIONS 3



struct RandomExplorationAlgorithm {

	RandomExplorationAlgorithm( peoPopEval< Route >& __popEval, peoSynchronousMultiStart< Route >& extParallelExecution ) 
		: popEval( __popEval ), parallelExecution( extParallelExecution ) { 
	}


	// the sequential algorithm to be executed in parallel by the wrapper
	void operator()() {

		RouteInit route_init;	// random init object - creates random Route objects
		RouteEval route_eval;
		eoPop< Route > population( RANDOM_POP_SIZE, route_init );

		popEval( population );


		// executing HCs on the population in parallel
		parallelExecution( population );



		// just to show off :: HCs on a vector of Route objects
		{
			Route* rVect = new Route[ 5 ];
			for ( unsigned int index = 0; index < 5; index++ ) {
	
				route_init( rVect[ index ] ); route_eval( rVect[ index ] );
			}
	
			// applying the HCs on the vector of Route objects
			parallelExecution( rVect, rVect + 5 );
			delete[] rVect;
		}



		Route bestRoute = population.best_element();

		for ( unsigned int index = 0; index < RANDOM_ITERATIONS; index++ ) {

			for ( unsigned int routeIndex = 0; routeIndex < RANDOM_POP_SIZE; routeIndex++ ) {

				route_init( population[ routeIndex ] );
			}

			popEval( population );

			if ( fabs( population.best_element().fitness() ) < fabs( bestRoute.fitness() ) ) bestRoute = population.best_element();

			std::cout << "Random Iteration #" << index << "... [ " << bestRoute.fitness() << " ]" << std::flush << std::endl; 
		}
	}


	peoPopEval< Route >& popEval;
	peoSynchronousMultiStart< Route >& parallelExecution;
};




int main( int __argc, char** __argv ) {

	srand( time(NULL) );



	// initializing the ParadisEO-PEO environment
	peo :: init( __argc, __argv );


	// processing the command line specified parameters
	loadParameters( __argc, __argv );



	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	// #1 An EO evolutionary algorithm to be executed in parallel with other algorithms (no parallel evaluation, no etc.).

	// init, eval operators, EA operators -------------------------------------------------------------------------------------------------------------
	RouteInit route_init;	// random init object - creates random Route objects
	RouteEval full_eval;	// evaluator object - offers a fitness value for a specified Route object

	OrderXover crossover;	// crossover operator - creates two offsprings out of two specified parents
	CitySwap mutation;	// mutation operator - randomly mutates one gene for a specified individual
	// ------------------------------------------------------------------------------------------------------------------------------------------------


	// evolutionary algorithm components --------------------------------------------------------------------------------------------------------------

	eoPop< Route > population( POP_SIZE, route_init );	// initial population for the algorithm having POP_SIZE individuals

	eoGenContinue< Route > eaCont( NUM_GEN );		// continuation criterion - the algorithm will iterate for NUM_GEN generations
	eoCheckPoint< Route > eaCheckpointContinue( eaCont );	// checkpoint object - verify at each iteration if the continuation criterion is met

	eoRankingSelect< Route > selectionStrategy;		// selection strategy - applied at each iteration for selecting parent individuals
	eoSelectNumber< Route > eaSelect( selectionStrategy, POP_SIZE ); // selection object - POP_SIZE individuals are selected at each iteration

	// transform operator - includes the crossover and the mutation operators with a specified associated rate
	eoSGATransform< Route > transform( crossover, CROSS_RATE, mutation, MUT_RATE );

	eoPlusReplacement< Route > eaReplace;			// replacement strategy - for replacing the initial population with offspring individuals
	// ------------------------------------------------------------------------------------------------------------------------------------------------



	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// SEQENTIAL ALGORITHM DEFINITION -----------------------------------------------------------------------------------------------------------------
	eoEasyEA< Route > eaAlg( eaCheckpointContinue, full_eval, eaSelect, transform, eaReplace );
	// SEQENTIAL ALGORITHM DEFINITION -----------------------------------------------------------------------------------------------------------------

	// SETTING UP THE PARALLEL WRAPPER ----------------------------------------------------------------------------------------------------------------
	peoParallelAlgorithmWrapper parallelEAAlg( eaAlg, population );	// specifying the embedded algorithm and the algorithm input data
	// SETTING UP THE PARALLEL WRAPPER ----------------------------------------------------------------------------------------------------------------
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<





	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	// #2 A MO hill climbing to be executed in parallel with other algorithms (no parallel evaluation, no etc.).

	if ( getNodeRank() == 1 ) {
		
		Graph::load( __argv [ 1 ] );
	}
	
	Route route;
	RouteInit init; init( route );
	RouteEval full_evalHC; full_evalHC( route );
	
	if ( getNodeRank() == 1 ) {

		std :: cout << "[From] " << route << std :: endl;
	}
	

	TwoOptInit two_opt_init;
	TwoOptNext two_opt_next;
	TwoOptIncrEval two_opt_incr_eval;
	
	moBestImprSelect< TwoOpt > two_opt_select;



	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// SEQENTIAL ALGORITHM DEFINITION -----------------------------------------------------------------------------------------------------------------
	moHC< TwoOpt > hill_climbing( two_opt_init, two_opt_next, two_opt_incr_eval, two_opt_select, full_evalHC );
	// SEQENTIAL ALGORITHM DEFINITION -----------------------------------------------------------------------------------------------------------------

	// SETTING UP THE PARALLEL WRAPPER ----------------------------------------------------------------------------------------------------------------
	peoParallelAlgorithmWrapper parallelHillClimbing( hill_climbing, route );	// specifying the embedded algorithm and the algorithm input data
	// SETTING UP THE PARALLEL WRAPPER ----------------------------------------------------------------------------------------------------------------
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<





	
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	// #3 A user defined algorithm to be executed in parallel with other algorithms - parallel evaluation and synchronous 
	//		multi-start of several hill-climbing algorithms (inside the user defined algorithm)!!.

	RouteEval full_evalRandom;
	peoParaPopEval< Route > randomParaEval( full_evalRandom );


	peoSynchronousMultiStart< Route > parallelExecution( hill_climbing );

	RandomExplorationAlgorithm randomExplorationAlgorithm( randomParaEval, parallelExecution );


	// SETTING UP THE PARALLEL WRAPPER ----------------------------------------------------------------------------------------------------------------
	peoParallelAlgorithmWrapper parallelRandExp( randomExplorationAlgorithm );	// specifying the embedded algorithm - no input data in this case

	randomParaEval.setOwner( parallelRandExp );
	parallelExecution.setOwner( parallelRandExp );
	// SETTING UP THE PARALLEL WRAPPER ----------------------------------------------------------------------------------------------------------------
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	// #4 Synchronous Multi-Start: several hill-climbing algorithms launched in parallel on different initial solutions

	RouteInit ex_hc_route_init;	// random init object - creates random Route objects
	RouteEval ex_hc_full_eval;	// evaluator object - offers a fitness value for a specified Route object

	eoPop< Route > ex_hc_population( POP_SIZE, ex_hc_route_init );

	for ( unsigned int index = 0; index < POP_SIZE; index++ ) {

		ex_hc_full_eval( ex_hc_population[ index ] );
	}


	// SETTING UP THE PARALLEL WRAPPER ----------------------------------------------------------------------------------------------------------------
	peoSynchronousMultiStart< Route > ex_hc_parallelExecution( hill_climbing );
	peoParallelAlgorithmWrapper ex_hc_parallel( ex_hc_parallelExecution, ex_hc_population );	// specifying the embedded algorithm - no input data in this case

	ex_hc_parallelExecution.setOwner( ex_hc_parallel );
	// SETTING UP THE PARALLEL WRAPPER ----------------------------------------------------------------------------------------------------------------
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<





	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	// #5 Synchronous Multi-Start: Multiple EO evolutionary algorithms to be executed in parallel
	//	(inside different processes, on different populations; no parallel evaluation, no etc.).

	RouteInit ex_route_init;	// random init object - creates random Route objects
	RouteEval ex_full_eval;		// evaluator object - offers a fitness value for a specified Route object

	std::vector< eoPop< Route > > ex_population;
	ex_population.resize( NUMBER_OF_POPULATIONS );

	for ( unsigned int indexPop = 0; indexPop < NUMBER_OF_POPULATIONS; indexPop++ ) {

		ex_population[ indexPop ].resize( POP_SIZE );

		for ( unsigned int index = 0; index < POP_SIZE; index++ ) {

			ex_route_init( ex_population[ indexPop ][ index ] );
			ex_full_eval( ex_population[ indexPop ][ index ] );
		}
	}


	// SETTING UP THE PARALLEL WRAPPER ----------------------------------------------------------------------------------------------------------------
	peoSynchronousMultiStart< eoPop< Route > > ex_parallelExecution( eaAlg );
	peoParallelAlgorithmWrapper ex_parallel( ex_parallelExecution, ex_population );	// specifying the embedded algorithm - no input data in this case

	ex_parallelExecution.setOwner( ex_parallel );
	// SETTING UP THE PARALLEL WRAPPER ----------------------------------------------------------------------------------------------------------------
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<






	peo :: run( );
	peo :: finalize( );
	// shutting down the ParadisEO-PEO environment



	// the algorithm is executed in the #1 rank process
	if ( getNodeRank() == 1 ) {

		std :: cout << "[To] " << route << std :: endl << std::endl;


		std :: cout << "Synchronous Multi-Start HCs:" << std :: endl ;
		for ( unsigned int index = 0; index < POP_SIZE; index++ ) {
	
			std::cout << ex_hc_population[ index ] << std::endl;
		}
		std::cout << std::endl << std::endl;


		std :: cout << "Synchronous Multi-Start EAs:" << std :: endl ;
		for ( unsigned int index = 0; index < NUMBER_OF_POPULATIONS; index++ ) {
	
			std::cout << ex_population[ index ] << std::endl;
		}
		std::cout << std::endl << std::flush;

	}



	return 0;
}
