//-----------------------------------------------------------------------------
// t-eogeneration.cpp
//-----------------------------------------------------------------------------

// to avoid long name warnings
#pragma warning(disable:4786)

#include <eo>

#include "binary_value.h"

//-----------------------------------------------------------------------------

typedef eoBin<float> Chrom;

//-----------------------------------------------------------------------------

main()
{
  const unsigned POP_SIZE = 8, CHROM_SIZE = 16;
  unsigned i;

  eoUniform<Chrom::Type> uniform(false, true);
  eoBinRandom<Chrom> random;
  eoPop<Chrom> pop; 
  
  for (i = 0; i < POP_SIZE; ++i)
    {
      Chrom chrom(CHROM_SIZE);
      random(chrom);
      binary_value(chrom);
      pop.push_back(chrom);
    }
  
  cout << "population:" << endl;
  for (i = 0; i < pop.size(); ++i)
    cout << "\t" << pop[i] << " " << pop[i].fitness() << endl;

  
  // selection
  eoLottery<Chrom> lottery;

  // breeder
  eoBinBitFlip<Chrom> bitflip;
  eoBinCrossover<Chrom> xover;
  eoProportionalOpSel<Chrom> propSel;
  eoBreeder<Chrom> breeder( propSel );
  propSel.addOp(bitflip, 0.25);
  propSel.addOp(xover, 0.75);
  
  // replacement
  eoInclusion<Chrom> inclusion;

  // Evaluation
  eoEvalFuncPtr<Chrom> eval(  binary_value );

  // GA generation
  eoGeneration<Chrom> generation(lottery, breeder, inclusion, eval);

  // evolution
  unsigned g = 0;
  do {
    try
      {
	generation(pop);
      }
    catch (exception& e)
      {
	cout << "exception: " << e.what() << endl;;
	exit(EXIT_FAILURE);
      }
    
    cout << "pop[" << ++g << "]" << endl;
    for (i = 0; i < pop.size(); ++i)
      cout << "\t" <<  pop[i] << " " << pop[i].fitness() << endl;
    
  } while (pop[0].fitness() < pow(2.0, CHROM_SIZE) - 1);

  return 0;
}

//-----------------------------------------------------------------------------
