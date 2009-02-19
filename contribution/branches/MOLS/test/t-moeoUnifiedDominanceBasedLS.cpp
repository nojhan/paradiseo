#include <eo>
#include <moeo>
#include <moeoPopNeighborhoodExplorer.h>
#include <moeoAllSolAllNeighborsExpl.h>
#include <moeoOneSolAllNeighborsExpl.h>
#include <moeoOneSolOneNeighborExpl.h>
#include <moeoAllSolOneNeighborExpl.h>
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
		eoInit<FlowShop>& init = do_make_genotype(parser, state);
	    // population
	    eoPop<FlowShop>& pop = do_make_pop(parser, state, init);
	    
	    eoTimeContinue < FlowShop > continuator(10000000);
	    moeoAllSolOneNeighborExpl < ExchangeMove > explorer(moveInit,moveNext, eval);
//	    
	    moeoUnifiedDominanceBasedLS < ExchangeMove > algo(continuator, explorer);
	   
	    for (unsigned int i=0; i<pop.size(); i++)
	    	eval(pop[i]);
        std::cout << "Initial Population\n";
        pop.sortedPrintOn(std::cout);
        std::cout << std::endl;
	    
//
	    algo(pop);

	std::cout << "OK c'est bon" << std::endl;
	return EXIT_SUCCESS;
}
