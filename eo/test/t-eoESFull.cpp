// Program to test several EO-ES features

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif 

#include <string>
#include <iostream>
#include <iterator>

using namespace std;

#include <utils/eoParser.h>

// evolution specific
#include <eoEvalFuncPtr.h>

// representation specific
#include <es/eoESFullChrom.h>            // though contained in following
//#include <eoESReco.h>
//#include <eoESMut.h>
//#include <eoESRandomize.h>
// this fitness
#include "real_value.h"		// the sphere fitness

// Now the main 
/////////////// 
typedef eoESFullChrom<float> Ind;
  
main(int argc, char *argv[]) {
//  unsigned mu, lambda;
//  bool comma;

  // Create the command-line parser
  eoParser parser( argc, argv, "Basic EA for vector<float> with adaptive mutations");

  // Define Parameters and load them
  eoValueParam<uint32>& seed        = parser.createParam(time(0), "seed", "Random number seed");
  eoValueParam<string>& load_name   = parser.createParam("", "Load","Load a state file",'L');
  eoValueParam<string>& save_name   = parser.createParam("", "Save","Saves a state file",'S');
 
  eoState state;
  state.registerObject(parser);
 
   if (load_name.value() != "")
   { // load the parser. This is only neccessary when the user wants to 
     // be able to change the parameters in the state file by hand.
       state.load(load_name.value()); // load the parser
   }

 
  // Evaluation
  eoEvalFuncPtr<Ind> eval(  real_value );



  /*
  // Evolution and population parameters
  eoScheme<Ind> the_scheme(parser);

  // recombination and mutation operators, reading their parameters from the parser
  eoESReco<float> MyReco(parser, FirstEO);
  eoESMutate<float> MyMut(parser, FirstEO);

  // termination conditions read by the parser
  eoTermVector<Ind> the_terms(parser);
  
  // Initialization of the population
  // shoudl be called using the parser, in case you want to read from file(s)
  eoESRandomize<float> randomize;	// an eoESInd randomnizer
  eoPop<Ind> pop(the_scheme.PopSize(), FirstEO, randomize); 
  // eval(pop);    // shoudl we call it from inside the constructor???

  // ALL parmeters have been read: write them out
  //  Writing the parameters on arv[0].status 
  // but of course this can be modified - see the example parser.cpp
  parser.outputParam();
  // except the help parameter???
  if( parser.getBool("-h" , "--help" , "Shows this help")) {
    parser.printHelp();
    exit(1);
  }

  unsigned i, iind;


    cout << "Initial population: \n" << endl;
    for (i = 0; i < pop.size(); ++i) {
      eval(pop[i]);
      cout << pop[i].fitness() << "\t" << pop[i] << endl;
    }

  // the Operators
  eoSequentialOpHolder <Ind> seqholder;
  //  seqholder.addOp(MyReco, 1.0);
  seqholder.addOp(MyMut, 1.0);

  // One generation
  eoEvolStep<Ind> evol_scheme(the_scheme, seqholder, eval);
  
  // the algorithm: 
  eoFullEA<Ind> ea(evol_scheme, the_terms);

  ea(pop);
  
    cout << "Final population: \n" << endl;
  for (i = 0; i < pop.size(); ++i)
    cout << pop[i].fitness() << "\t" << pop[i] << endl;
	*/
	return 0;  
}


