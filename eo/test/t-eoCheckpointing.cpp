//-----------------------------------------------------------------------------

// to avoid long name warnings
#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#include <stdexcept>  // runtime_error 

//-----------------------------------------------------------------------------
// tt.cpp: 
//
//-----------------------------------------------------------------------------


// general
#include <utils/eoRNG.h>		// Random number generators
#include <ga/eoBin.h>
#include <utils/eoParser.h>
#include <utils/eoState.h>

//-----------------------------------------------------------------------------

// include package checkpointing
#include <utils/checkpointing>
#include <eoGenTerm.h>

struct Dummy : public EO<double>
{
    typedef double Type;
};


struct eoDummyPop : public eoPop<Dummy>
{
public :
    eoDummyPop(int s = 2) { resize(s); }
};

//-----------------------------------------------------------------------------

int the_main(int argc, char **argv)
{ // ok, we have a command line parser and a state
  
    typedef eoBin<float> Chrom;

    eoParser parser(argc, argv);
      
    // Define Parameters
    eoValueParam<unsigned>& chrom_size = parser.createParam(unsigned(2), "chrom-size", "Chromosome size");
    eoValueParam<double> rate(0.01, "mutationRatePerBit", "Initial value for mutation rate per bit"); 
    eoValueParam<double> factor(0.99, "mutationFactor", "Decrease factor for mutation rate");
    eoValueParam<uint32> seed(time(0), "seed", "Random number seed");
    eoValueParam<string> load_name("", "Load","Load",'L');
    eoValueParam<string> save_name("", "Save","Save",'S');
 
    // Register them
    parser.processParam(rate,       "Genetic Operators");
    parser.processParam(factor,     "Genetic Operators");
    parser.processParam(load_name,  "Persistence");
    parser.processParam(save_name,  "Persistence");
    parser.processParam(seed,       "Rng seeding");

   eoState state;
   state.registerObject(parser);
 
   if (load_name.value() != "")
   { // load the parser. This is only neccessary when the user wants to 
     // be able to change the parameters in the state file by hand.
       state.load(load_name.value()); // load the parser
   }
  
    // Create the algorithm here
    typedef Dummy EoType;

    eoDummyPop pop;
    
    eoGenTerm<EoType> genTerm(5); // 5 generations

    eoCheckPoint<EoType> checkpoint(genTerm);

    eoValueParam<unsigned> generationCounter(0, "Generation");
    eoIncrementor<unsigned> increment(generationCounter.value());

    checkpoint.add(increment);

    eoFileMonitor monitor("monitor.csv");
    checkpoint.add(monitor);

    monitor.add(generationCounter);

    eoSecondMomentStats<EoType> stats;

    checkpoint.add(stats);
    monitor.add(stats);

    eoCountedStateSaver stateSaver1(3, state, "generation"); // save every third generation
    eoTimedStateSaver   stateSaver2(2, state, "time"); // save every 2 seconds

    checkpoint.add(stateSaver1);
    checkpoint.add(stateSaver2);

    // Register the algorithm
    state.registerObject(rng);
    state.registerObject(pop);

    if (parser.userNeedsHelp())
    {
        parser.printHelp(cout);
        return 0;
    }

    // Either load or initialize
    if (load_name.value() != "")
    {
        state.load(load_name.value()); // load the rest
    }
    else
    {
        // else

        // initialize rng and population

        rng.reseed(seed.value());
        
        pop.resize(2);

        pop[0].fitness(1);
        pop[1].fitness(2);
    }

    while(checkpoint(pop))
    {
        pop[0].fitness(pop[0].fitness() + 1);
        
        time_t now = time(0);

        while (time(0) == now) {} // wait a second to test timed saver

        cout << "gen " << generationCounter.value() << endl;
    }

    // run the algorithm

    // Save when needed
    if (save_name.value() != "")
    {
        string file_name = save_name.value();
        save_name.value() = ""; // so that it does not appear in the parser section of the state file
        state.save(file_name);
    }

    for (int i = 0; i < 100; ++i)
        rng.rand();

    cout << "a random number is " << rng.random(1024) << endl;;

    return 1;
}

int main(int argc, char **argv)
{
    try
    {
        the_main(argc, argv);
    }
    catch(exception& e)
    {
        cout << "Exception: " << e.what() << endl;
    }

    return 1;
}