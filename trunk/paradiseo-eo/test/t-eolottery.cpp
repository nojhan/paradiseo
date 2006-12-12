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
  
  std::cout << "original population:" << std::endl;
  sort(pop.begin(), pop.end());
  for (i = 0; i < pop.size(); i++)
    std::cout << pop[i] << "  " << pop[i].fitness() << std::endl;
  
  eoLottery<Chrom> lottery;
  lottery(pop, pop2); 

  std::cout << "selected by lottery population:" << std::endl;
  sort(pop2.begin(), pop2.end());
  for (i = 0; i < pop2.size(); i++)
    std::cout << pop2[i] << "  " << pop2[i].fitness() << std::endl;

  return 0;
}

//-----------------------------------------------------------------------------
