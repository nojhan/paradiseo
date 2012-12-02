#include <smp>
#include <eo>

#include "smpTestClass.h"

using namespace paradiseo::smp;
using namespace std;

int main(void)
{
    typedef struct {
        unsigned popSize = 10;
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

    eoGenContinue<Indi> genCont(param.maxGen+100); // generation continuation
    eoGenContinue<Indi> genCont_2(param.maxGen); // generation continuation
    eoGenContinue<Indi> genCont_3(param.maxGen); // generation continuation
    
    // Define population
    eoPop<Indi> pop(param.popSize, chromInit);

    try
    {
        // Island 1
        // // Emigration policy
        // // // Element 1 
        eoPeriodicContinue<Indi> criteria(5);
        eoDetTournamentSelect<Indi> selectOne(20);
        eoSelectNumber<Indi> who(selectOne, 3);
        
        MigPolicy<Indi> migPolicy;
        migPolicy.push_back(PolicyElement<Indi>(who, criteria));
        
        // // Integration policy
        eoPlusReplacement<Indi> intPolicy;

        eoPop<Indi> pop(param.popSize, chromInit);
        
        Island<eoEasyEA,Indi> test(pop, intPolicy, migPolicy, genCont, plainEval, select, transform, replace);
        
        // Island 2
        // // Emigration policy
        // // // Element 1 
        eoPeriodicContinue<Indi> criteria_2(5);
        eoDetTournamentSelect<Indi> selectOne_2(25);
        eoSelectNumber<Indi> who_2(selectOne_2, 5);
        
        MigPolicy<Indi> migPolicy_2;
        migPolicy_2.push_back(PolicyElement<Indi>(who_2, criteria_2));
        
        // // Integration policy
        eoPlusReplacement<Indi> intPolicy_2;
        
        eoPop<Indi> pop2(30, chromInit);
        Island<eoEasyEA,Indi> test2(pop2, intPolicy_2, migPolicy_2, genCont_2, plainEval, select, transform, replace);
          
        // Island 3
        // // Emigration policy
        // // // Element 1 
        eoPeriodicContinue<Indi> criteria_3(10);
        eoDetTournamentSelect<Indi> selectOne_3(15);
        eoSelectNumber<Indi> who_3(selectOne_3, 1);
        
        MigPolicy<Indi> migPolicy_3;
        migPolicy.push_back(PolicyElement<Indi>(who_3, criteria_3));
        
        // // Integration policy
        eoPlusReplacement<Indi> intPolicy_3;
        
        eoPop<Indi> pop3(30, chromInit);
        Island<eoEasyEA,Indi> test3(pop3, intPolicy_3, migPolicy_3, genCont_3, plainEval, select, transform, replace);
        
        // Topology
        Topology<Complete> topo;
        
        IslandModel<Indi> model(topo);
        model.add(test);
        model.add(test2);
        model.add(test3);
        
        model();
        
        cout << test.getPop() << endl;
        cout << test2.getPop() << endl;
        cout << test3.getPop() << endl;
    }
    catch(exception& e)
    {
      cout << "Exception: " << e.what() << '\n';
    }
    
    return 0;
}
