#include <smp>
#include <eo>
#include <ga.h>

#include "smpTestClass.h"

using namespace paradiseo::smp;
using namespace std;

typedef eoBit<double> Indi2; // A bitstring with fitness double

// Conversion functions
Indi2 fromBase(Indi& i, unsigned size)
{
    (void)i;
    // Dummy conversion. We just create a new Indi2
    Indi2 v;
    for (unsigned ivar=0; ivar<size; ivar++)
	{
	      bool r = rng.flip(); // new value, random in {0,1}
	      v.push_back(r);      // append that random value to v
	}
    std::cout << "Convert from base : " << v << std::endl;
    return v;
}

Indi toBase(Indi2& i)
{
    (void)i;
    // Dummy conversion. We just create a new Indi
    Indi v;
    std::cout << "Convert to base : " << v << std::endl;
    return v;
}

// Eval function for the PSO
// A simple fitness function that computes the number of ones of a bitstring
// @param _Indi2 A biststring Indi2vidual

double binary_value(const Indi2 & _Indi2)
{
    double sum = 0;
    for (unsigned i = 0; i < _Indi2.size(); i++)
        sum += _Indi2[i];
    return sum;
}

int main(void)
{
//////////////////////////////////////////////////////////////////
// PSO PART
//////////////////////////////////////////////////////////////////
    // PSO general parameters
    const unsigned int SEED = 42;      // seed for random number generator
    const unsigned int T_SIZE = 3;     // size for tournament selection
    const unsigned int VEC_SIZE = 16;   // Number of bits in genotypes
    const unsigned int POP_SIZE = 10;  // Size of population
    const unsigned int MAX_GEN = 10;  // Maximum number of generation before STOP
    const float CROSS_RATE = 0.8;      // Crossover rate
    const double P_MUT_PER_BIT = 0.01; // probability of bit-flip mutation
    const float MUT_RATE = 1.0;        // mutation rate

    rng.reseed(SEED);

    eoEvalFuncPtr<Indi2> eval(binary_value);

    // PSO population initialization
    eoPop<Indi2> pop;

    for(unsigned int igeno=0; igeno<POP_SIZE; igeno++)
    {
        Indi2 v;           // void Indi2vidual, to be filled
        for (unsigned ivar=0; ivar<VEC_SIZE; ivar++)
        {
            bool r = rng.flip(); // new value, random in {0,1}
            v.push_back(r);      // append that random value to v
        }
        eval(v);                 // evaluate it
        pop.push_back(v);        // and put it in the population
    }

    // ISLAND 1 : PSO
    // // Algorithm part
    eoDetTournamentSelect<Indi2> select(T_SIZE);  // T_SIZE in [2,POP_SIZE]
    eo1PtBitXover<Indi2> xover;
    eoBitMutation<Indi2> mutation(P_MUT_PER_BIT);
    eoGenContinue<Indi2> continuator(MAX_GEN);

    // // Emigration policy
    // // // Element 1 
    eoPeriodicContinue<Indi2> criteria(1);
    eoDetTournamentSelect<Indi2> selectOne(2);
    eoSelectNumber<Indi2> who(selectOne, 1);
        
    MigPolicy<Indi2> migPolicy;
    migPolicy.push_back(PolicyElement<Indi2>(who, criteria));
        
    // // Integration policy
    eoPlusReplacement<Indi2> intPolicy;
        
    // We bind conversion functions
    auto frombase = std::bind(fromBase, std::placeholders::_1, VEC_SIZE);
    auto tobase = std::bind(toBase, std::placeholders::_1);

    Island<eoSGA,Indi2, Indi> gga(frombase, tobase, pop, intPolicy, migPolicy, select, xover, CROSS_RATE, mutation, MUT_RATE, eval, continuator);
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// EasyEA PART
//////////////////////////////////////////////////////////////////
    // EA general parameters
    typedef struct {
        unsigned popSize = 10;
        unsigned tSize = 2;
        double pCross = 0.8;
        double pMut = 0.7;
        unsigned maxGen = 10;
    } Param; 

    Param param;
    loadInstances("t-data.dat", n, bkv, a, b);
      
    // Evaluation function
    IndiEvalFunc plainEval;
    
    // Init a solution
    IndiInit chromInit;
    
    // Define selection
    eoDetTournamentSelect<Indi> selectOne2(param.tSize);
    eoSelectPerc<Indi> select2(selectOne2);// by default rate==1

    // Define operators for crossover and mutation
    IndiXover Xover;                 // CROSSOVER
    IndiSwapMutation  mutationSwap;  // MUTATION
      
    // Encapsule in a tranform operator
    eoSGATransform<Indi> transform(Xover, param.pCross, mutationSwap, param.pMut);
    
    // Define replace operator
    eoPlusReplacement<Indi> replace;
    eoGenContinue<Indi> genCont(param.maxGen); // generation continuation
    eoPop<Indi> pop2(param.popSize, chromInit);

    // ISLAND 2 : EasyEA
    // // Emigration policy
    // // // Element 1 
    eoPeriodicContinue<Indi> criteria2(1);
    eoDetTournamentSelect<Indi> selectOne3(5);
    eoSelectNumber<Indi> who2(selectOne3, 2);
        
    MigPolicy<Indi> migPolicy2;
    migPolicy2.push_back(PolicyElement<Indi>(who2, criteria2));
        
    // // Integration policy
    eoPlusReplacement<Indi> intPolicy2;

    Island<eoEasyEA,Indi> test(pop2, intPolicy2, migPolicy2, genCont, plainEval, select2, transform, replace);
    
    // MODEL CREATION
    Topology<Complete> topo;
    IslandModel<Indi> model(topo);
    
    try
    {
    
        model.add(test);
        model.add(gga);
            
        model();
            
        cout << test.getPop() << endl;
        cout << gga.getPop() << endl;
        
    }
    catch(exception& e)
    {
        cout << "Exception: " << e.what() << '\n';
    }    

    return 0;
}
