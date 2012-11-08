#include <smp.h>
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
        unsigned maxGen = 100;
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
        // Emigration policy
        eoPeriodicContinue<Indi> criteria(25); // We mig each gen
        Policy<Indi> pol;
        pol.push_back(PolicyElement<Indi>(select, criteria));
        
        Island<eoEasyEA,Indi> test(param.popSize, chromInit, replace, pol, genCont, plainEval, select, transform, replace);
        test();
        cout << test.getPop() << endl;
    }
    catch(exception& e)
    {
      cout << "Exception: " << e.what() << '\n';
    }
    
    return 0;
}
