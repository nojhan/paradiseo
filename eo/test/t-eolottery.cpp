//-----------------------------------------------------------------------------
// t-eolottery.cpp
//-----------------------------------------------------------------------------

#include <eo>  // eoBin, eoPop, eoLottery

//-----------------------------------------------------------------------------

typedef eoBin<float> Chrom;

void binary_value(Chrom& chrom)
{
  float sum = 0; 
  for (unsigned i = 0; i < chrom.size(); i++)
    if (chrom[i])
      sum += pow(2, i);
  chrom.fitness(sum);
}

//-----------------------------------------------------------------------------

main()
{
  const unsigned POP_SIZE = 8, CHROM_SIZE = 4;

  eoPop<Chrom> pop, pop2;
  eoBinRandom<Chrom> random;
  unsigned i;

  for (i = 0; i < POP_SIZE; i++)
    {
      Chrom chrom(CHROM_SIZE);
      random(chrom);
      binary_value(chrom);
      pop.push_back(chrom);
    }
  
  cout << "original population:" << endl;
  sort(pop.begin(), pop.end());
  for (i = 0; i < pop.size(); i++)
    cout << pop[i] << "  " << pop[i].fitness() << endl;
  
  eoLottery<Chrom> lottery;
  lottery(pop, pop2); 

  cout << "selected by lottery population:" << endl;
  sort(pop2.begin(), pop2.end());
  for (i = 0; i < pop2.size(); i++)
    cout << pop2[i] << "  " << pop2[i].fitness() << endl;

  return 0;
}

//-----------------------------------------------------------------------------
