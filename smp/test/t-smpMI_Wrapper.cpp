#include <smp>
#include <eo>

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

    try
    {
        
        std::vector<eoPop<Indi>> pops = IslandModelWrapper<eoEasyEA,Indi>(50, topo, 100, chromInit,  intPolicy, migPolicy, genCont, plainEval, select, transform, replace);
        
        for(auto& pop : pops)
        {
            for(auto& indi : pop)
                plainEval(indi);
            pop.sort();
            std::cout << pop << std::endl;
        }        

    }
    catch(exception& e)
    {
      cout << "Exception: " << e.what() << '\n';
    }
    
    return 0;
}
