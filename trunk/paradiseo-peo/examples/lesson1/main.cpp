// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "main_ga.cpp"

// (c) OPAC Team, LIFL, January 2006

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "param.h"
#include "route_init.h"
#include "route_eval.h"

#include "order_xover.h"
#include "city_swap.h"

#include <paradiseo>

#define POP_SIZE 10
#define NUM_GEN 100
#define CROSS_RATE 1.0
#define MUT_RATE 0.01


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
	peoSeqPopEval< Route > eaPopEval( full_eval );		// evaluator object - to be applied at each iteration on the entire population

	eoGenContinue< Route > eaCont( NUM_GEN );		// continuation criterion - the algorithm will iterate for NUM_GEN generations
	eoCheckPoint< Route > eaCheckpointContinue( eaConts );	// checkpoint object - verify at each iteration if the continuation criterion is met

	eoRankingSelect< Route > selectionStrategy;		// selection strategy - applied at each iteration for selecting parent individuals
	eoSelectNumber< Route > eaSelect( selectionStrategy, POP_SIZE ); // selection object - POP_SIZE individuals are selected at each iteration

	// transform operator - includes the crossover and the mutation operators with a specified associated rate
	eoSGATransform< Route > transform( crossover, CROSS_RATE, mutation, MUT_RATE );
	peoSeqTransform< Route > eaTransform( transform );	// ParadisEO transform operator (please remark the peo prefix) - wraps an e EO transform object

	eoPlusReplacement< Route > eaReplace;			// replacement strategy - for replacing the initial population with offspring individuals
	// ------------------------------------------------------------------------------------------------------------------------------------------------


	// ParadisEO-PEO evolutionary algorithm -----------------------------------------------------------------------------------------------------------

	peoEA< Route > eaAlg( eaCheckpointContinue, eaPopEval, eaSelect, eaTransform, eaReplace );
	
	eaAlg( population );	// specifying the initial population for the algorithm, to be iteratively evolved
	// ------------------------------------------------------------------------------------------------------------------------------------------------


	peo :: run( );
	peo :: finalize( );
	// shutting down the ParadisEO-PEO environment

	return 0;
}
