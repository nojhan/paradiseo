//eo
#include <eo>
#include <eoSwapMutation.h>

//general
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <unistd.h>

// moeo
#include <moeo>
#include <do/make_continue_moeo.h>
#include <do/make_checkpoint_moeo.h>

// fsp
#include <fsp>

//peo
#include <peo>
#include <peoAsyncDataTransfer.h>
#include <peoSyncDataTransfer.h>
#include <core/star_topo.h>
#include <peoParallelAlgorithmWrapper.h>
#include <core/eoVector_mesg.h>

//DMLS


#include <moeoTransfer.h>

void make_help(eoParser & _parser);

using namespace std;

int main(int argc, char* argv[])
{
  try{

    eoParser parser(argc, argv);  // for user-parameter reading
    eoState state;                // to keep all things allocated     
    
    /*** number of objectives ***/
    fspObjectiveVectorTraits::nObj = 2;
    
    /*** parameters ***/
    eoValueParam<uint32_t>& _seedParam = parser.createParam(uint32_t(0), "seed", "Random number seed", 'S');
    std::string _file = parser.createParam(std::string(), "file", "", '\0', "Representation", true).value();
    double _crossRate = parser.createParam((double) 0.05, "crossRate", "Rate for 2-PT crossover", 0, "Variation Operators").value();
    double _mutRate = parser.createParam((double) 1.0, "mutRate", "Rate for shift mutation", 0, "Variation Operators").value();
    unsigned int _popSize = parser.createParam((unsigned int)(100), "popSize", "Population Size", 'P', "Evolution Engine").value();
    std::string _strategy = parser.createParam(std::string(), "explorer", "OneOne - OneAll - AllOne - AllAll - OneFirst - AllFirst - OneND - AllND", '\0', "Evolution Engine", true).value();
    unsigned int _nbKick = parser.createParam((unsigned int)(10), "nbKick", "Number of kick", 'K', "Evolution Engine").value();  
    
    // seed
    if (_seedParam.value() == 0)
      _seedParam.value() = time(0);
    rng.reseed(_seedParam.value());
    
    /*** the representation-dependent things ***/
    
    // load data
    fspData data(_file);
    // size
    unsigned int size = data.getN(); // nb jobs
    // init
    fspInit init(size, 0);
    // eval
    fspEval simpleEval(data.getM(), data.getN(), data.getP(), data.getD());
    eoEvalFuncCounter<FSP> eval(simpleEval);
    // cross
    fspCross cross;
    // mut
    fspMut mut;
    // op
    eoSGAGenOp<FSP> op(cross, _crossRate, mut, _mutRate);
    
    /*** the representation-independent things ***/
  
        // move init
    fspMoveInit moveInit;
    // move next
    fspMoveNext moveNext;
    // move incr eval
    fspMoveIncrEval moveIncrEval(data.getM(), data.getN(), data.getP(), data.getD());


    bool multiply=true;
    moeoPopNeighborhoodExplorer<fspMove> * explorer;
    moeoUnvisitedSelect<FSP> * selector;
    if (_strategy == std::string("OneOne"))
      {
	selector = new moeoNumberUnvisitedSelect<FSP> (1);
	explorer = new moeoSimpleSubNeighborhoodExplorer<fspMove> (moveInit, moveNext, moveIncrEval, 1);	
      }
    else if (_strategy == std::string("OneAll"))
      {
	selector = new moeoNumberUnvisitedSelect<FSP> (1);
	explorer = new moeoExhaustiveNeighborhoodExplorer<fspMove> (moveInit, moveNext, moveIncrEval);
      }
    else if (_strategy == std::string("AllOne"))
      {
	selector = new moeoExhaustiveUnvisitedSelect<FSP>;
	explorer = new moeoSimpleSubNeighborhoodExplorer<fspMove> (moveInit, moveNext, moveIncrEval, 1);
	multiply=false;
      }
    else if (_strategy == std::string("AllAll"))
      {
	selector = new moeoExhaustiveUnvisitedSelect<FSP>;
	explorer = new moeoExhaustiveNeighborhoodExplorer<fspMove> (moveInit, moveNext, moveIncrEval);
      }
    else if (_strategy == std::string("OneFirst"))
      {
	selector = new moeoNumberUnvisitedSelect<FSP> (1);
	explorer = new moeoFirstImprovingNeighborhoodExplorer<fspMove> (moveInit, moveNext, moveIncrEval);
      }
    else if (_strategy == std::string("AllFirst"))
      {
	selector = new moeoExhaustiveUnvisitedSelect<FSP>;
	explorer = new moeoFirstImprovingNeighborhoodExplorer<fspMove> (moveInit, moveNext, moveIncrEval);
      }
    else if (_strategy == std::string("OneND"))
      {
	selector = new moeoNumberUnvisitedSelect<FSP> (1);
	explorer = new moeoNoDesimprovingNeighborhoodExplorer<fspMove> (moveInit, moveNext, moveIncrEval);
      }
    else if (_strategy == std::string("AllND"))
      {
	selector = new moeoExhaustiveUnvisitedSelect<FSP>;
	explorer = new moeoNoDesimprovingNeighborhoodExplorer<fspMove> (moveInit, moveNext, moveIncrEval);
	multiply=false;
      }
    else
      {
	std::string stmp = std::string("Invalid explorer strategy: ") + _strategy;
	throw std::runtime_error(stmp.c_str());
      }
    state.storeFunctor(selector);
    state.storeFunctor(explorer);

    //peo :: init (argc, argv);

    eoPop<FSP> popIBEA;
    eoPop<FSP> popLS;
    eoPop<FSP> popCentral;
    eoPop<FSP> popCentral2;
    popCentral.resize(0);
    popCentral2.resize(0);
    // initialization of the population
    popIBEA.append(_popSize, init);
    popLS.append(1, init);
    // definition of an unbounded archive
    moeoNewBoundedArchive<FSP> archCentral(100);
    moeoNewBoundedArchive<FSP> archCentral2(100);
    moeoNewBoundedArchive<FSP> archLS(100);
    moeoUnboundedArchive<FSP> archIBEA;

    // stopping criteria
    eoContinue<FSP> & stop = do_make_continue_moeo(parser, state, eval);

    eoGenContinue<FSP> stop1(5000000);
    eoGenContinue<FSP> stop2(5000000);
    eoGenContinue<FSP> stop3(5000000);
 
    // checkpointing
    eoCheckPoint<FSP> & checkpoint1 = do_make_checkpoint_moeo(parser, state, eval, stop1, popIBEA, archIBEA);
    eoCheckPoint<FSP> & checkpoint2 = do_make_checkpoint_moeo(parser, state, eval, stop2, popLS, archLS);
    eoCheckPoint<FSP> & checkpoint3 = do_make_checkpoint_moeo(parser, state, eval, stop3, popCentral, archCentral);

    moeoArchiveObjectiveVectorSavingUpdater < FSP > * save_updater1 = new moeoArchiveObjectiveVectorSavingUpdater < FSP > (archIBEA, "archIBEA");
    state.storeFunctor(save_updater1);
    checkpoint1.add(*save_updater1);

    moeoArchiveObjectiveVectorSavingUpdater < FSP > * save_updater2 = new moeoArchiveObjectiveVectorSavingUpdater < FSP > (archLS, "archLS");
    state.storeFunctor(save_updater2);
    checkpoint2.add(*save_updater2);

    moeoArchiveObjectiveVectorSavingUpdater < FSP > * save_updater3 = new moeoArchiveObjectiveVectorSavingUpdater < FSP > (archCentral, "archCentral");
    state.storeFunctor(save_updater3);
    checkpoint3.add(*save_updater3);

    // metric
    moeoAdditiveEpsilonBinaryMetric<fspObjectiveVector> metric;
    // algorithms
    moeoIBEA<FSP> algo1 (checkpoint1, eval, op, metric);
    //moeoSEEA<FSP> algo1 (checkpoint1, eval, op, arch1);
    moeoUnifiedDominanceBasedLS <fspMove> algo2(checkpoint2, eval, archLS, *explorer, *selector);

    //moeoUnifiedDominanceBasedLS <tspMove> algo3(checkpoint3, eval, arch3, *explorer, *selector);
    //moeoSEEA<FSP> algo3 (checkpoint3, eval, op, arch3);
    //moeoSEEA<FSP> algo4 (checkpoint4, eval, op, arch4);
    //moeoIBEA<FSP> algo3 (checkpoint3, eval, op, metric);
    //moeoNSGAII<FSP> algo3 (checkpoint3, eval, op);
    
    //PEO:initialisation 


    
    //Topolgy
    RingTopology ring1, ring2;
    StarTopology star;




    eoSwapMutation <FSP> swap;



    CentralAggregation<FSP> test;
    IBEAAggregation<FSP> test2;
    LSAggregation<FSP> test3(swap, eval, _nbKick);
    ArchToArchAggregation<FSP> test4;

	apply<FSP>(eval, popIBEA);
	popIBEA.sort();
	apply<FSP>(eval, popLS);
	popLS.sort();

	

	/*    peoSyncDataTransfer transfer1(archCentral, ring1, test);
    centralArchive<FSP> centre1(archCentral, popCentral, transfer1, checkpoint3);

    
    peoSyncDataTransfer transfer2(archCentral, ring2, test);
    centralArchive<FSP> centre2(archCentral, popCentral, transfer2, checkpoint3);



    
    peoSyncDataTransfer transfer3(popIBEA, ring1, test2);
    eoGenContinue <FSP> cont2(500);

    gestionTransfer<FSP> gest2(cont2, transfer3, archIBEA);
    //centralArchive centre1(arch1, pop1, transfer1);
    checkpoint1.add(gest2);

    

    
    peoSyncDataTransfer transfer4(archLS, ring2, test3);
    eoAmeliorationContinue <FSP> cont(archLS, 1, false);
    //eoGenContinue <FSP> cont(10000);
    gestionTransfer2<FSP> gest(cont, transfer4, archLS);
    checkpoint2.add(gest);*/


	//  std::vector<peoSyncDataTransfer> vect;


    //    initDebugging();
    //setDebugMode(true);
    
	testPEO<FSP> hop(argc, argv, algo1, algo2, popIBEA, popLS, archCentral, test, test, test2, test3, checkpoint1, checkpoint2, checkpoint3);


    hop();


    
    // Start the parallel EA
    /*eoPop<FSP> dummyPop;
    dummyPop.resize(0);

    eoPop<FSP> dummyPop2;
    dummyPop2.resize(0);*/

    //Wrapp algorithms
    
    /*      peoParallelAlgorithmWrapper parallelEA_1(centre1, dummyPop);
      transfer1.setOwner( parallelEA_1 );
    
    
    peoParallelAlgorithmWrapper parallelEA_2(algo1, popIBEA);
    transfer3.setOwner( parallelEA_2 );
    
    
    peoParallelAlgorithmWrapper parallelEA_3(centre2, dummyPop2);
    transfer2.setOwner( parallelEA_3 );
    
    
    peoParallelAlgorithmWrapper parallelEA_4(algo2, popLS);
    transfer4.setOwner( parallelEA_4 );*/

    
    /*    if (getNodeRank()==1)
      {

	cout << "Initial Population IBEA\n" << popIBEA << endl;
      } 
    
    if (getNodeRank()==2)
      {

	cout << "Initial Population LS\n" << popLS << endl;
	}*/

    

    
    
    //run
    //peo :: run( );
    //peo :: finalize( );
    //endDebugging();
  


  }
  
  catch (exception& e){
    cout << e.what() << endl;
  }
  return EXIT_SUCCESS;
}
