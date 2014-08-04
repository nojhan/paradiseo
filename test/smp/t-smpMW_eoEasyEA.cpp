#include <paradiseo/smp/MWModel.h>
#include <cassert>
#include <paradiseo/eo.h>

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
        unsigned maxGen = 1000;
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
    
    // Define population
    eoPop<Indi> pop(param.popSize, chromInit);

    try
    {
        
        MWModel<eoEasyEA,Indi> mw(genCont, plainEval, select, transform, replace);
        mw.evaluate(pop);
        std::cout << "Initial population :" << std::endl;
        pop.sort();
        std::cout << pop << std::endl;
        mw(pop);
        std::cout << "Final population :" << std::endl;
        pop.sort();
        std::cout << pop << std::endl;
    }
    catch(exception& e)
    {
      cout << "Exception: " << e.what() << '\n';
    }
    
    assert(pop.nth_element_fitness(0) == 229092);
    
    return 0;
}
