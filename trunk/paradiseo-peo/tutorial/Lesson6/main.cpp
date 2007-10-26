/* 
* <main.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
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
*          clive.canape@inria.fr
*/

#include "route.h"
#include "route_init.h"
#include "route_eval.h"

#include "order_xover.h"
#include "city_swap.h"

#include "param.h"

#include "merge_route_eval.h"
#include "part_route_eval.h"


#include <peo>


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

	/*
	 * You can change the size of the population thanks POP_SIZE
	 * 
	 */

	eoPop< Route > population( POP_SIZE, route_init );	

	/* 
	 * If you want to use a parallel evaluation :  peoParaPopEval< Route > eaPopEval( full_eval );
	 * Else, you can use a sequential evaluation : peoSeqPopEval< Route > eaPopEval( full_eval );
	 * 
	 */
	peoParaPopEval< Route > eaPopEval( full_eval );		

	/*
	 * Continuation criterion 
	 */
	eoGenContinue< Route > eaCont( NUM_GEN );		
	eoCheckPoint< Route > eaCheckpointContinue( eaCont );	
	
	/*
	 * Selection strategy
	 */
	eoRankingSelect< Route > selectionStrategy;		
	eoSelectNumber< Route > eaSelect( selectionStrategy, POP_SIZE ); 
	
	/* With this transform operator, you can use a parallel crossover and a parallel mutation
	 * 
	 * Unfortunately, if you don't use a crossover which creates two children with two parents,
	 * you can't use this operator.
	 * In this case, you should send a mail to : paradiseo-help@lists.gforge.inria.fr
	 * 
	 */
	peoParaSGATransform< Route > eaTransform( crossover, CROSS_RATE, mutation, MUT_RATE );	// ParadisEO transform operator (please remark the peo prefix) - wraps an e EO transform object

	/*
	 * Replacement strategy
	 */
	eoPlusReplacement< Route > eaReplace;			
	

	// ParadisEO-PEO evolutionary algorithm -----------------------------------------------------------------------------------------------------------

	peoEA< Route > eaAlg( eaCheckpointContinue, eaPopEval, eaSelect, eaTransform, eaReplace );
	
	eaAlg( population );	// specifying the initial population for the algorithm, to be iteratively evolved
	// ------------------------------------------------------------------------------------------------------------------------------------------------


	peo :: run( );
	peo :: finalize( );
	
	if(getNodeRank()==1)
		std::cout<<"\n\nPopulation :\n"<<population;

	return 0;
}
