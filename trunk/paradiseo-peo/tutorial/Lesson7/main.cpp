#include <peo>
#include <moeo>
#include <make_eval_FlowShop.h>
#include <make_genotype_FlowShop.h>
#include <make_op_FlowShop.h>
#include <do/make_pop.h>
#include <../tutorial/Lesson7/make_continue_moeo.h>
#include <../tutorial/Lesson7/make_checkpoint_moeo.h>
#include <../tutorial/Lesson7/make_ea_moeo.h>
#include <../tutorial/Lesson7/make_para_eval.h>
#include <FlowShop.h>



int main(int argc, char* argv[])
{

	  peo :: init( argc,argv );
      eoParser parser(argc, argv);  
      eoState state;                
      peoMoeoPopEval<FlowShop>& eval = do_make_para_eval(parser, state);
      eoInit<FlowShop>& init = do_make_genotype(parser, state);
      eoGenOp<FlowShop>& op = do_make_op(parser, state);
      eoPop<FlowShop>& pop = do_make_pop(parser, state, init);
      moeoArchive<FlowShop> arch;
      eoContinue<FlowShop>& term = do_make_continue_moeo(parser, state, eval);
      eoCheckPoint<FlowShop>& checkpoint = do_make_checkpoint_moeo(parser, state, eval, term, pop, arch);
      eoAlgo<FlowShop>& algo = do_make_ea_moeo(parser, state, eval, checkpoint, op, arch);
      peoWrapper parallelMOEO( algo, pop);
      eval.setOwner(parallelMOEO);
      peo :: run();
      peo :: finalize();
      if (getNodeRank()==1)
   	  	std::cout << "Final population :\n" << pop << std::endl;
}
