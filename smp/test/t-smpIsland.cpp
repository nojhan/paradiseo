#include <smp.h>
#include <eo>

#include "smpTestClass.h"

using namespace paradiseo::smp;
using namespace std;

int main(void)
{
    typedef struct {
        unsigned popSize = 3;
        unsigned tSize = 2;
        double pCross = 0.8;
        double pMut = 0.7;
        unsigned maxGen = 10;
    } Param; 

    Param param;
    
    rng.reseed(42);

    loadInstances("t-data.dat", n, bkv, a, b);
      
    // Evaluation function
    IndiEvalFunc plainEval;
    
    // Init a solution
    IndiInit chromInit;
    
    // Define selection
    eoDetTournamentSelect<Indi> selectOne(param.tSize);
    eoSelectPerc<Indi> select(selectOne);// by default rate==1

    // Define operators for crossover and mutation
    IndiXover Xover;                 // CROSSOVER
    IndiSwapMutation  mutationSwap;  // MUTATION
      
    // Encapsule in a tranform operator
    eoSGATransform<Indi> transform(Xover, param.pCross, mutationSwap, param.pMut);
    
    // Define replace operator
    eoPlusReplacement<Indi> replace;

    eoGenContinue<Indi> genCont(param.maxGen); // generation continuation
    eoGenContinue<Indi> genCont_2(param.maxGen); // generation continuation
    
    // Define population
    eoPop<Indi> pop(param.popSize, chromInit);

    try
    {
        // Island 1
        // // Emigration policy
        // // // Element 1 
        eoPeriodicContinue<Indi> criteria(5);
        eoDetTournamentSelect<Indi> selectOne(2);
        eoSelectNumber<Indi> who(selectOne, 1);
        
        MigPolicy<Indi> migPolicy;
        migPolicy.push_back(PolicyElement<Indi>(who, criteria));
        
        // // Integration policy
        eoPlusReplacement<Indi> intPolicy;

        Island<eoEasyEA,Indi> test(param.popSize, chromInit, intPolicy, migPolicy, genCont, plainEval, select, transform, replace);
        
        // Island 2
        // // Emigration policy
        // // // Element 1 
        eoPeriodicContinue<Indi> criteria_2(5);
        eoDetTournamentSelect<Indi> selectOne_2(2);
        eoSelectNumber<Indi> who_2(selectOne_2, 1);
        
        MigPolicy<Indi> migPolicy_2;
        migPolicy_2.push_back(PolicyElement<Indi>(who_2, criteria_2));
        
        // // Integration policy
        eoPlusReplacement<Indi> intPolicy_2;
  
        Island<eoEasyEA,Indi> test2(param.popSize, chromInit, intPolicy_2, migPolicy_2, genCont_2, plainEval, select, transform, replace);
        
        // Topology
        Topology<Complete> topo;
        
        IslandModel<Indi> model(topo);
        model.add(test);
        model.add(test2);
        
        model();
        
        cout << test.getPop() << endl;
        cout << test2.getPop() << endl;
    }
    catch(exception& e)
    {
      cout << "Exception: " << e.what() << '\n';
    }
    
    return 0;
}
