//-----------------------------------------------------------------------------
// t-eobreeder.cpp
//-----------------------------------------------------------------------------

// to avoid long name warnings
#pragma warning(disable:4786)

#include <eoBin.h>  // eoBin, eoPop, eoBreeder
#include <eoPop.h>
#include <eoBitOp.h>
#include <eoProportionalOpSel.h>
#include <eoBreeder.h>

//-----------------------------------------------------------------------------

typedef eoBin<float> Chrom;

//-----------------------------------------------------------------------------

void binary_value(Chrom& chrom)
{
  float sum = 0;
  for (unsigned i = 0; i < chrom.size(); i++)
    if (chrom[i])
      sum += pow(2, chrom.size() - i - 1);
  chrom.fitness(sum);
}

//-----------------------------------------------------------------------------

main()
{
  const unsigned POP_SIZE = 8, CHROM_SIZE = 4;
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
    cout << pop[i] << " " << pop[i].fitness() << endl;

  eoBinBitFlip<Chrom> bitflip;
  eoBinCrossover<Chrom> xover;
  eoProportionalOpSel<Chrom> propSel;
  eoBreeder<Chrom> breeder( propSel );
  propSel.addOp(bitflip, 0.25);
  propSel.addOp(xover, 0.75);

  breeder(pop);

  // reevaluation of fitness 
  for_each(pop.begin(), pop.end(), binary_value);

  cout << "new population:" << endl;
  for (i = 0; i < pop.size(); ++i)
    cout << pop[i] << " " << pop[i].fitness() << endl;

  return 0;
}

//-----------------------------------------------------------------------------
