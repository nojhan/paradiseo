#include <stdexcept>
#include <iostream>
#include <strstream>

#include <paradiseo.h>
#include <ga.h>

typedef eoBit<double> Indi ;	// A bitstring with fitness double

#include "binary_value.h"

void main_function(int argc, char **argv) {

  // Some parameters
  const unsigned int SEED = 42 ; // Seed for random number generator
  const unsigned int T_SIZE = 3 ; // Size for tournament selection
  const unsigned int VEC_SIZE = 8 ; // Number of bits in genotypes
  const unsigned int POP_SIZE = 100 ; // Size of population

  const unsigned int MAX_GEN = 20 ; // Maximum number of generation before STOP

  const double P_CROSS = 0.8 ; // Crossover probability
  const double P_MUT = 1.0 ; // Mutation probability
  const double P_MUT_PER_BIT = 0.01 ; // Internal probability for bit-flip mutation
  const double onePointRate = 0.5 ; // Rate for 1-pt Xover
  const double bitFlipRate = 0.5 ; // Rate for bit-flip mutation

  rng.reseed (SEED) ;

  eoEvalFuncPtr <Indi, double, const vector <bool>& > eval (binary_value) ;

  eoUniformGenerator <bool> uGen ;
  eoInitFixedLength <Indi> random (VEC_SIZE, uGen) ; 
  
  eoPop <Indi> pop (POP_SIZE, random) ;

  apply <Indi> (eval, pop) ;

  eoDetTournamentSelect <Indi> selectOne (T_SIZE) ;

  eoSelectPerc<Indi> select (selectOne) ;

  eoGenerationalReplacement <Indi> replace ; 

  eo1PtBitXover<Indi> xover1 ;
  
  eoBitMutation<Indi>  mutationBitFlip(P_MUT_PER_BIT) ;
  
  // The operators are  encapsulated into an eoTRansform object
  eoSGATransform<Indi> transform(xover1, P_CROSS, mutationBitFlip, P_MUT);

  eoGenContinue<Indi> genCont (MAX_GEN);
  
  eoEasyEA<Indi> gga (genCont, eval, select, transform, replace);

  eoListener <Indi> listen (argc, argv) ;
  
  eoDistEvalEasyEA <Indi> dist_gga (listen, gga, "Mars") ;
  
  dist_gga (pop) ;
 
  listen.destroy ("Mars") ;
  
  // OUTPUT
  // Print (sorted) intial population
  pop.sort();
  cout << "FINAL Population\n" << pop << endl;
// GENERAL
}

// A main that catches the exceptions

int main(int argc, char **argv)
{

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
