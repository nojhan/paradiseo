#include <iostream>

#include <ga/ga.h>
#include "binary_value.h"
#include <apply.h>

using namespace std;

int main(int argc, char* argv[])
{

  try
  {
  typedef eoBit<double> EoType;

  eoParser parser(argc, argv);

  eoState state;         // keeps all things allocated, including eoEasyEA and eoPop!

  eoEvalFuncPtr<EoType, float> eval( binary_value<EoType> );
  eoGenContinue<EoType>  term(20);
  eoCheckPoint<EoType>   checkpoint(term);

  eoAlgo<EoType>& ga = make_ga(parser, eval, checkpoint, state);

  eoPop<EoType>& pop   = init_ga(parser, state, double());

  if (parser.userNeedsHelp())
  {
    parser.printHelp(cout);
    return 0;
  }

  apply(eval, pop);

  run_ga(ga, pop); // run the ga
  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
}