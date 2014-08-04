#include <paradiseo/smp.h>
#include <paradiseo/eo.h>

#include "smpTestClass.h"

using namespace paradiseo::smp;
using namespace std;

int main(void)
{
    // Defining parameters
    typedef struct {
        unsigned popSize = 1000;
        unsigned tSize = 2;
        double pCross = 0.8;
        double pMut = 0.7;
        unsigned maxGen = 1000;
    } Param; 

    Param param;
    
    // Fixing the seed
    rng.reseed(42);

    // Load instance
    loadInstances("t-data.dat", n, bkv, a, b);
      
    //Common part to all islands
    IndiEvalFunc plainEval;
    IndiInit chromInit;
    eoDetTournamentSelect<Indi> selectOne(param.tSize);
    eoSelectPerc<Indi> select(selectOne);// by default rate==1
    IndiXover Xover;                 // CROSSOVER
    IndiSwapMutation  mutationSwap;  // MUTATION
    eoSGATransform<Indi> transform(Xover, param.pCross, mutationSwap, param.pMut);
    eoPlusReplacement<Indi> replace;
    
    // MODEL
    // Topologies
    Topology<Complete> topo;
    IslandModel<Indi> model(topo);
    
    // ISLAND 1
    // // Algorithm part
    eoGenContinue<Indi> genCont(param.maxGen+100);
    eoPop<Indi> pop(param.popSize, chromInit);
    // // Emigration policy
    // // // Element 1
    eoPeriodicContinue<Indi> criteria(5);
    eoDetTournamentSelect<Indi> selectOne1(20);
    eoSelectNumber<Indi> who(selectOne1, 3);
    
    MigPolicy<Indi> migPolicy;
    migPolicy.push_back(PolicyElement<Indi>(who, criteria));
        
    // // Integration policy
    eoPlusReplacement<Indi> intPolicy;

    Island<eoEasyEA,Indi> test(pop, intPolicy, migPolicy, genCont, plainEval, select, transform, replace);

    // ISLAND 1
    // // Algorithm part
    eoGenContinue<Indi> genCont_2(param.maxGen); // generation continuation
    eoPop<Indi> pop2(30, chromInit);
    // // Emigration policy
    // // // Element 1 
    eoPeriodicContinue<Indi> criteria_2(5);
    eoDetTournamentSelect<Indi> selectOne_2(25);
    eoSelectNumber<Indi> who_2(selectOne_2, 5);
        
    MigPolicy<Indi> migPolicy_2;
    migPolicy_2.push_back(PolicyElement<Indi>(who_2, criteria_2));
        
    // // Integration policy
    eoPlusReplacement<Indi> intPolicy_2;
         
    Island<eoEasyEA,Indi> test2(pop2, intPolicy_2, migPolicy_2, genCont_2, plainEval, select, transform, replace);
    
    // Island 3
    // // Algorithm part
    eoGenContinue<Indi> genCont_3(param.maxGen);
    eoPop<Indi> pop3(30, chromInit);
    // // Emigration policy
    // // // Element 1 
    eoPeriodicContinue<Indi> criteria_3(10);
    eoDetTournamentSelect<Indi> selectOne_3(15);
    eoSelectNumber<Indi> who_3(selectOne_3, 1);
        
    MigPolicy<Indi> migPolicy_3;
    migPolicy.push_back(PolicyElement<Indi>(who_3, criteria_3));
        
    // // Integration policy
    eoPlusReplacement<Indi> intPolicy_3;
    
    Island<eoEasyEA,Indi> test3(pop3, intPolicy_3, migPolicy_3, genCont_3, plainEval, select, transform, replace);
    

    try
    {

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
