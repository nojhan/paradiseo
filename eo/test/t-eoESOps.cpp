// Program to test several EO-ES features

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif 

#include <string>
#include <iostream>
#include <iterator>

using namespace std;

// general
#include <utils/eoParser.h>            // though contained in all others!
// evolution specific
#include <eoEvalFuncPtr.h>
//#include <eoSequentialOpHolder.h>
//#include <eoFullEA.h>
// representation specific

#include <es/eoESFullChrom.h>            // though contained in following
//#include <eoESReco.h>
#include <es/eoESFullMut.h>
//#include <eoESRandomize.h>
// this fitness
#include "real_value.h"		// the sphere fitness

// Now the main 
/////////////// 
typedef eoESFullChrom<float> Ind;
  
main(int argc, char *argv[]) {
  unsigned mu, lambda;
  bool comma;

  // Create the command-line parser
  Parser parser( argc, argv, "Basic EA for vector<float> with adaptive mutations");

  //reproducible random seed - thanks, Maarten
  InitRandom(parser);

  // a first Ind, reading its parameters from the parser
  // will be used later to inialize the whole population
  Ind FirstEO(parser);

  // Evaluation
  // here we should call some  parser-based constructor, 
  // as many evaluation function need parameters
  // and also have some preliminary stuffs to do
  eoEvalFuncPtr<Ind> eval(  real_value );

 // recombination and mutation operators, reading their parameters from the parser
  eoESMutate<float> MyMut(parser, 
			  FirstEO.StdDevLength(), FirstEO.size(), 
			  FirstEO.CorCffLength() );

  std::cout << "First EO " << FirstEO << std::endl;
  MyMut(FirstEO);
  std::cout << "First EO mutated" << FirstEO << std::endl;

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


    std::cout << "Initial population: \n" << std::endl;
    for (i = 0; i < pop.size(); ++i) {
      eval(pop[i]);
      std::cout << pop[i].fitness() << "\t" << pop[i] << std::endl;
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
  
    std::cout << "Final population: \n" << std::endl;
  for (i = 0; i < pop.size(); ++i)
    std::cout << pop[i].fitness() << "\t" << pop[i] << std::endl;
  return 0;
  */
}

