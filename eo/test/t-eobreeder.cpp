//-----------------------------------------------------------------------------
// t-eobreeder.cpp
//-----------------------------------------------------------------------------

#include <stdlib.h>  // srand
#include <time.h>    // time
#include <eo>        // eoBin, eoPop, eoBreeder

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
  srand(time(NULL));

  const unsigned POP_SIZE = 8, CHROM_SIZE = 4;
  unsigned i;

  eoUniform<Chrom::Type> uniform(false, true);
  eoBinRandom<Chrom> random;
  eoPop<Chrom> pop, pop2; 
  
  for (i = 0; i < POP_SIZE; i++)
    {
      Chrom chrom(CHROM_SIZE);
      random(chrom);
      binary_value(chrom);
      pop.push_back(chrom);
    }
  
  eoBinBitFlip<Chrom> bitflip;
  eoBinCrossover<Chrom> xover;
  eoBreeder<Chrom> breeder;
  breeder.add(bitflip, 1.0);
  breeder.add(xover, 1.0);

  pop2 = pop;
  breeder(pop2);
   
  for (i = 0; i < pop2.size(); i++)
    binary_value(pop2[i]);

  cout << "population: \tnew population" << endl;
  for (i = 0; i < pop.size(); i++)
    cout << pop[i] << " " << pop[i].fitness() << "     \t"
	 << pop2[i] << " " << pop2[i].fitness() << endl;

  return 0;
}

//-----------------------------------------------------------------------------
