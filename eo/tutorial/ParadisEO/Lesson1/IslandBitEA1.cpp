#include <paradiseo.h>
#include <ga.h>

typedef eoBit <double> Indi;	// A bitstring with fitness double

#include "binary_value.h"

void main_function(int argc, char ** argv) {
  
  // Some parameters ...
  const unsigned int T_SIZE = 3 ; // Size for tournament selection
  const unsigned int VEC_SIZE = 50 ; // Number of bits in genotypes
  const unsigned int POP_SIZE = 100 ; // Size of population

  const unsigned int MAX_GEN = 1000 ; // Fixed number of generations

  const double P_CROSS = 0.8 ; // Crossover probability
  const double P_MUT = 1.0 ; // Mutation probability

  const double P_MUT_PER_BIT = 0.01 ; // Internal probability for bit-flip mutation
  const double onePointRate = 0.5 ; // Rate for 1-pt Xover
  const double bitFlipRate = 0.5 ; // Rate for bit-flip mutation

  eoEvalFuncPtr <Indi, double, const vector <bool> & > eval (binary_value) ;
  eoUniformGenerator <bool> uGen ;
  eoInitFixedLength <Indi> random (VEC_SIZE, uGen) ;

  eoPop <Indi> pop (POP_SIZE, random) ;

  apply <Indi> (eval, pop) ; // A first evaluation of the population

  eoDetTournamentSelect <Indi> selectOne(T_SIZE) ;
  eoSelectPerc <Indi> select (selectOne) ; // The selection operator
  
  eoGenerationalReplacement<Indi> replace ; // The replacement operator

  eo1PtBitXover <Indi> xover1 ;
  eoPropCombinedQuadOp <Indi> xover (xover1, onePointRate) ;
  eoBitMutation <Indi> mutationBitFlip (P_MUT_PER_BIT) ;
  eoPropCombinedMonOp <Indi> mutation (mutationBitFlip, bitFlipRate) ;

  eoSGATransform <Indi> transform (xover, P_CROSS, mutation, P_MUT) ;

  eoGenContinue<Indi> genCont (MAX_GEN) ; // The continuation criteria
  
  // First evolutionnary algorithm
  eoEasyEA <Indi> gga (genCont, eval, select, transform, replace) ;
  
  // What's new ?
  eoListener <Indi> listen (argc, argv) ;
  rng.reseed (listen.here ().number ()) ; 
 
  vector <string> v ;
  v.push_back ("Mars1") ;
  v.push_back ("Mars2") ;
  eoRingConnectivity <Indi> conn (listen, v) ; // The ring topology used
  
  eoCyclicGenContinue <Indi> cycl_cont (300) ; // Immigration step all 300 evolutions
  eoRandomSelect <Indi> sel_rand ; // Random selection of emigrants
  eoSelectMany <Indi> sel (sel_rand, 0.1) ; /* How many individuals should be selected
					       to be sent ? */
  eoPlusReplacement <Indi> repl ; // How to integrate new individuals ?
  // A island esay evolutionnary named "Mars"
  eoIslandsEasyEA <Indi> islgga ("Mars1", listen, conn, gga, cycl_cont, sel, repl) ;
  islgga (pop) ;
  pop.sort () ;
  cout << "The final population is now ..." << endl ;
  cout << pop << endl ;
}

int main(int argc, char **argv) {
#ifdef _MSC_VER

  int flag = _CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF);
  flag |= _CRTDBG_LEAK_CHECK_DF;
  _CrtSetDbgFlag(flag);

#endif
  
  try {
    main_function(argc, argv) ;
  }
  catch(exception& e) {
    cout << "Exception: " << e.what () << '\n' ;
  }

  return 1 ;
}
