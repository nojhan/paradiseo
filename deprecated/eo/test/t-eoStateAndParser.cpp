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
#include <ga.h>
#include <utils/eoParser.h>
#include <utils/eoState.h>

//-----------------------------------------------------------------------------

// include package checkpointing
#include <utils/checkpointing>
// and provisions for Bounds reading
#include <utils/eoRealVectorBounds.h>

struct Dummy : public EO<double>
{
    typedef double Type;
};


//-----------------------------------------------------------------------------

int the_main(int argc, char **argv)
{ // ok, we have a command line parser and a state

    typedef eoBit<float> Chrom;

    eoParser parser(argc, argv);

    // Define Parameters
    eoValueParam<unsigned int> dimParam((unsigned int)(5), "dimension", "dimension");
    eoValueParam<double> rate(0.01, "mutationRatePerBit", "Initial value for mutation rate per bit");
    eoValueParam<double> factor(0.99, "mutationFactor", "Decrease factor for mutation rate");
    eoValueParam<uint32_t> seed(time(0), "seed", "Random number seed");
    // test if user entered or if default value used
    if (parser.isItThere(seed))
      std::cout << "YES\n";
    else
      std::cout << "NO\n";

    eoValueParam<std::string> load_name("", "Load","Load",'L');
    eoValueParam<std::string> save_name("", "Save","Save",'S');


    // Register them
    parser.processParam(dimParam,   "Genetic Operators");
    parser.processParam(rate,       "Genetic Operators");
    parser.processParam(factor,     "Genetic Operators");
    parser.processParam(load_name,  "Persistence");
    parser.processParam(save_name,  "Persistence");
    parser.processParam(seed,       "Rng seeding");

    // a bound param (need dim)
    eoValueParam<eoRealVectorBounds> boundParam(eoRealVectorBounds(dimParam.value(),eoDummyRealNoBounds), "bounds","bounds",'b');

    parser.processParam(boundParam, "Genetic Operators");

    std::cout << "Bounds: " << boundParam.value() << std::endl;

   eoState state;
   state.registerObject(parser);


   if (load_name.value() != "")
   { // load the parser. This is only neccessary when the user wants to
     // be able to change the parameters in the state file by hand.
       state.load(load_name.value()); // load the parser
   }

    // Create the algorithm here

    // Register the algorithm
    state.registerObject(rng);
    //state.registerObject(pop);

    if (parser.userNeedsHelp())
    {
	parser.printHelp(std::cout);
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
    }

    // run the algorithm

    // Save when needed
    if (save_name.value() != "")
    {
	std::string file_name = save_name.value();
	save_name.value() = ""; // so that it does not appear in the parser section of the state file
	state.save(file_name);
    }

    for (int i = 0; i < 100; ++i)
	rng.rand();

    std::cout << "a random number is " << rng.random(1024) << std::endl;;

    return 1;
}

int main(int argc, char **argv)
{
    try
    {
	the_main(argc, argv);
    }
    catch(std::exception& e)
    {
	std::cout << "Exception: " << e.what() << std::endl;
    }

}
