#include <stdexcept>
#include <iostream>
#include <strstream>

#include <eo>
#include <ga.h>

typedef eoBit<double> Indi;

#include "binary_value.h"

void main_function(int argc, char **argv)
{

  const unsigned int SEED = 42;
  const unsigned int VEC_SIZE = 8;
  const unsigned int POP_SIZE = 25;

  const unsigned int MAX_GEN = 100; 

  const double P_MUT_PER_BIT = 0.01;

  rng.reseed(SEED);

  eoEvalFuncPtr<Indi, double, const vector<bool>& > eval(  binary_value );

  eoUniformGenerator<bool> uGen;
  eoInitFixedLength<Indi> random(VEC_SIZE, uGen);

  eoPop<Indi> pop(POP_SIZE, random);


  apply<Indi>(eval, pop);

  pop.sort();

  cout << "Initial Population" << endl;
  cout << pop;

  eo1PtBitXover<Indi> xover1;
 
  eoBitMutation<Indi>  mutationBitFlip(P_MUT_PER_BIT);

  eoGenContinue<Indi> genCont(MAX_GEN);
  
  eoBestSelect <Indi> select ;

  eoToricCellularEasyEA <Indi> gga (genCont,
				    eval,
				    select,
				    xover1,
				    mutationBitFlip,
				    select,
				    select) ;
  
  cout << "\n        Here we go\n\n";
  gga(pop);
  
  pop.sort();
  cout << "FINAL Population\n" << pop << endl;

}

// A main that catches the exceptions

int main(int argc, char **argv)
{
#ifdef _MSC_VER
  //  rng.reseed(42);
    int flag = _CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF);
     flag |= _CRTDBG_LEAK_CHECK_DF;
    _CrtSetDbgFlag(flag);
//   _CrtSetBreakAlloc(100);
#endif

    try
    {
        main_function(argc, argv);
    }
    catch(exception& e)
    {
        cout << "Exception: " << e.what() << '\n';
    }

    return 1;
}
