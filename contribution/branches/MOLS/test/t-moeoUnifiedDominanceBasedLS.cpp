#include <eo>
#include <moeo>
#include <moeoPopNeighborhoodExplorer.h>
#include <moeoAllSolAllNeighborsExpl.h>
#include <moeoPopLS.h>
#include <moeoUnifiedDominanceBasedLS.h>
#include <moMove.h>
#include <FlowShop.h>
#include <exchange_move.h>
#include <exchange_move_init.h>
#include <exchange_move_next.h>
#include <FlowShopEval.h>

// for the creation of an evaluator
#include <make_eval_FlowShop.h>
// for the creation of an initializer
#include <make_genotype_FlowShop.h>
// for the creation of the variation operators
#include <make_op_FlowShop.h>
// how to initialize the population
#include <do/make_pop.h>


int main(int argc, char* argv[])
{
	 	eoParser parser(argc, argv);  // for user-parameter reading
	    eoState state;                // to keep all things allocated
		ExchangeMoveNext moveNext;
		ExchangeMoveInit moveInit;
		ExchangeMove move;
		eoEvalFuncCounter<FlowShop>& eval = do_make_eval(parser, state);
		
	    // population
	    eoPop < FlowShop > pop;
	    eoTimeContinue < FlowShop > continuator(5);
	    moeoAllSolAllNeighborsExpl < ExchangeMove > explorer(moveInit,moveNext, eval);
//	    
	    moeoUnifiedDominanceBasedLS < ExchangeMove > algo(continuator, explorer);
	    
//
//	    algo(pop);

	std::cout << "OK c'est bon" << std::endl;
	return EXIT_SUCCESS;
}
