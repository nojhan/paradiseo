//-----------------------------------------------------------------------------
// t-eogeneration.cpp
//-----------------------------------------------------------------------------

// to avoid long name warnings
#pragma warning(disable:4786)

#include <eo>

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
  
  // GA generation
  eoGeneration<Chrom> generation(lottery, breeder, inclusion);

  // evolution
  unsigned g = 0;
  do {
    try
      {
	generation(pop, binary_value);
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
