//-----------------------------------------------------------------------------
// t-eoinclusion.cpp
//-----------------------------------------------------------------------------

#include <eo>  // eoBin, eoPop, eoInclusion

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
  const unsigned CHROM_SIZE = 4;
  unsigned i;

  eoUniform<Chrom::Type> uniform(false, true);
  eoBinRandom<Chrom> random;

  for (unsigned POP_SIZE = 4; POP_SIZE <=6; POP_SIZE++)
    {
      eoPop<Chrom> pop; 
      unsigned i;
      for ( i = 0; i < POP_SIZE; i++)
	    {
	      Chrom chrom(CHROM_SIZE);
	      random(chrom);
	      binary_value(chrom);
	      pop.push_back(chrom);
	    }
      
      for (unsigned POP2_SIZE = 4; POP2_SIZE <=6; POP2_SIZE++)
	{
	  eoPop<Chrom> pop2, pop3, pop4, pop5;
	 	
	  for (i = 0; i < POP2_SIZE; i++)
	    {
	      Chrom chrom(CHROM_SIZE);
	      random(chrom);
	      binary_value(chrom);
	      pop2.push_back(chrom);
	    }
	
	  cout << "--------------------------------------------------" << endl
	       << "breeders \tpop" << endl
	       << "--------------------------------------------------" << endl;
	  for (i = 0; i < max(pop.size(), pop2.size()); i++)
	    {	  
	      if (pop.size() > i) 
		cout << pop[i] << " " << pop[i].fitness() << "   \t";
	      else
		cout << "\t\t";
	      if (pop2.size() > i)
		cout << pop2[i] << " " << pop2[i].fitness();
	      cout << endl;
	    }
	
	  eoInclusion<Chrom> inclusion(0.75);
	  pop3 = pop2;
	  inclusion(pop, pop3); 
	  
	  eoInclusion<Chrom> inclusion2; 
	  pop4 = pop2;
	  inclusion2(pop, pop4); 
	  
	  eoInclusion<Chrom> inclusion3(1.5);
	  pop5 = pop2;
	  inclusion3(pop, pop5); 
	  
	  cout << endl
	       << "0.75 \t\t1.0 \t\t1.5" << endl
	       << "---- \t\t--- \t\t---" << endl;
	  for (i = 0; i < pop5.size(); i++)
	    {
	      if (pop3.size() > i)
		cout << pop3[i] << " " << pop3[i].fitness() << "   \t";
	      else
		cout << " \t\t";
	      if (pop4.size() > i)
		cout << pop4[i] << " " << pop4[i].fitness() << "   \t";
	      else
		cout << " \t\t";
	      if (pop5.size() > i)
		cout << pop5[i] << " " << pop5[i].fitness();
	      cout << endl;
	    }
	}
    }
  
  return 0;
}

//-----------------------------------------------------------------------------
