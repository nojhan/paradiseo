#include <peo> 
#include <moeo>
#include "parallelStruct.h"
#include <make_library.h>
#include <FlowShop.h>

int main(int argc, char* argv[])
{
	
  peo :: init( argc,argv );
  const unsigned int  MIG_FREQ = 10;
  const unsigned int  MIG_SIZE = 1;
  rng.reseed (time(0));
  
  
// Global archive
  peoMoeoArchive<FlowShop> globalArch;
  if (getNodeRank()==1)
  	std::cout << "Archive before :\n" << globalArch << std::endl;
  	
/***************************************************/  
/***************** First algorithm *****************/
/***************************************************/	

  // Topology	
  RingTopology topology;
  	
  // Algorithm
  eoParser parser(argc, argv);
  eoState state;
  peoMoeoPopEval<FlowShop>& eval = do_make_para_eval(parser, state);
  eoInit<FlowShop>& init = do_make_genotype(parser, state);
  eoGenOp<FlowShop>& op = do_make_op(parser, state);
  peoMoeoPop<FlowShop>& pop = do_make_pop(parser, state, init); // peoMoeoPop is defined in parallelStruct.h
  moeoArchive<FlowShop> arch;
  eoContinue<FlowShop>& term = do_make_continue_moeo(parser, state, eval);
  eoCheckPoint <  FlowShop> & checkpoint = state.storeFunctor(new eoCheckPoint < FlowShop > (term));
  
  // Communication between EA and the global archive
  		
  	// Selection mode in the EA
  		moeoRandomSelect <FlowShop> mig_select_one;
  		moeoSelector <FlowShop, peoMoeoPop<FlowShop> > mig_select (mig_select_one,MIG_SIZE,pop); // moeoSelector is defined in parallelStruct.h
  	// Mode of replacement in the EA	
  		moeoReplace < peoMoeoArchive<FlowShop>, peoMoeoPop<FlowShop> > mig_replace (pop);	// moeoReplace is defined in parallelStruct.h
  	// Continuator for the island
  		eoPeriodicContinue< FlowShop> mig_cont( MIG_FREQ );
  		eoContinuator<FlowShop> cont(mig_cont, pop);
  	// Communication : EA ---> global archive
  		peoAsyncIslandMig< peoMoeoPop<FlowShop>, peoMoeoArchive<FlowShop> > mig(cont,mig_select, mig_replace, topology);
  		checkpoint.add(mig);		
  	// Selection mode in the global archive
  		moeoSelectorArchive < peoMoeoArchive<FlowShop> > mig_selectArchive (globalArch); // moeoSelectorArchive is defined in parallelStruct.h
  	// Mode of replacement in the global archive
  		moeoReplaceArchive < peoMoeoPop<FlowShop>, peoMoeoArchive<FlowShop> > mig_replaceArchive (globalArch); // moeoReplaceArchive is defined in parallelStruct.h 		
  	// Continuator for the island
  		eoPeriodicContinue< FlowShop> mig_contArchive( MIG_FREQ );
  		eoContinuator<FlowShop> contArchive(mig_contArchive, pop);	
  	// Communication : global archive ---> EA
  		peoAsyncIslandMig< peoMoeoArchive<FlowShop>, peoMoeoPop<FlowShop> > migArchive(contArchive, mig_selectArchive, mig_replaceArchive, topology);
  		checkpoint.add(migArchive);
 
  eoAlgo<FlowShop>& algo = do_make_ea_moeo(parser, state, eval, checkpoint, op, arch);
  peoWrapper parallelMOEO( algo, pop);
  eval.setOwner(parallelMOEO);
  // Migrations
  mig.setOwner(parallelMOEO);
  migArchive.setOwner(parallelMOEO);

/***************************************************/
/***************************************************/
/***************************************************/
  
// The same one

  RingTopology topology2;
  eoParser parser2(argc, argv);
  eoState state2;
  peoMoeoPopEval<FlowShop>& eval2 = do_make_para_eval(parser2, state2);
  eoInit<FlowShop>& init2 = do_make_genotype(parser2, state2);
  eoGenOp<FlowShop>& op2 = do_make_op(parser2, state2);
  peoMoeoPop<FlowShop>& pop2 = do_make_pop(parser2, state2, init2); 
  moeoArchive<FlowShop> arch2;
  eoContinue<FlowShop>& term2 = do_make_continue_moeo(parser2, state2, eval2);
  eoCheckPoint <  FlowShop> & checkpoint2 = state2.storeFunctor(new eoCheckPoint < FlowShop > (term2));
  moeoRandomSelect <FlowShop> mig_select_one2;
  moeoSelector <FlowShop, peoMoeoPop<FlowShop> > mig_select2 (mig_select_one2,MIG_SIZE,pop2); 
  moeoReplace < peoMoeoArchive<FlowShop>, peoMoeoPop<FlowShop> > mig_replace2 (pop2);	
  eoPeriodicContinue< FlowShop> mig_cont2( MIG_FREQ );
  eoContinuator<FlowShop> cont2(mig_cont2, pop2);
  peoAsyncIslandMig< peoMoeoPop<FlowShop>, peoMoeoArchive<FlowShop> > mig2(cont2,mig_select2, mig_replace2, topology2);
  checkpoint2.add(mig2);		
  moeoSelectorArchive < peoMoeoArchive<FlowShop> > mig_selectArchive2 (globalArch); 
  moeoReplaceArchive < peoMoeoPop<FlowShop>, peoMoeoArchive<FlowShop> > mig_replaceArchive2 (globalArch); 
  eoPeriodicContinue< FlowShop> mig_contArchive2( MIG_FREQ );
  eoContinuator<FlowShop> contArchive2(mig_contArchive2, pop2);	
  peoAsyncIslandMig< peoMoeoArchive<FlowShop>, peoMoeoPop<FlowShop> > migArchive2(contArchive2, mig_selectArchive2, mig_replaceArchive2, topology2);
  checkpoint2.add(migArchive2);
  eoAlgo<FlowShop>& algo2 = do_make_ea_moeo(parser2, state2, eval2, checkpoint2, op2, arch2);
  peoWrapper parallelMOEO2( algo2, pop2);
  eval2.setOwner(parallelMOEO2);
  mig2.setOwner(parallelMOEO2);
  migArchive2.setOwner(parallelMOEO2);
  
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
    {
      pop.sort();
      std::cout << "Final population :\n" << pop << std::endl;
      pop2.sort();
      std::cout << "Final population :\n" << pop2 << std::endl;
      globalArch.sort();
      std::cout << "Archive after :\n" << globalArch << std::endl;
    }
}
