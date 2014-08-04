#include <cassert>
#include <vector>
#include <cstdlib>

#include <paradiseo/smp.h>
#include <paradiseo/eo.h>

#include "smpTestClass.h"

using namespace std;
using namespace paradiseo::smp;

int main(void)
{
    srand(time(NULL));

    loadInstances("t-data.dat", n, bkv, a, b);
    // Evaluation function
    IndiEvalFunc plainEval;
    // Init a solution
    IndiInit chromInit;
    
    eoPop<Indi> pop(100, chromInit);


    int nbWorkers = 4;

    Scheduler<Indi> sched(nbWorkers);
    
    sched(plainEval, pop);
    
    std::cout << pop << std::endl;
    
    // All indi would be evaluate once
    for( unsigned i = 0; i < pop.size(); i++)
        assert(pop[i].evalNb == 1);
  
    return 0;
}
