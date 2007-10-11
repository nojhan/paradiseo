//-----------------------------------------------------------------------------
// t-eoTwoOptMutation.cpp
//-----------------------------------------------------------------------------

#include <eo>

#include <eoInt.h>
#include <eoTwoOptMutation.h>

//-----------------------------------------------------------------------------

typedef eoInt<double> Chrom;

//-----------------------------------------------------------------------------


double real_value(const Chrom & _chrom)
{
  double sum = 0;
  for (unsigned i = 0; i < _chrom.size(); i++)
      sum += _chrom[i];
  return sum/_chrom.size();
}

// Return true if the given chromosome corresponds to a permutation
// There must be an nicer way to do it (set?) ...
bool check_permutation(const Chrom & _chrom)
{
	for (unsigned i = 0; i < _chrom.size(); ++i)
		for (unsigned j = 0; j < _chrom.size(); ++j)
			if(i!=j)
				if(_chrom[i]==_chrom[j]){
						 std::cout << " Error: Wrong permutation !" << std::endl;
						 std::string s;
						 s.append( " Wrong permutation in t-eoTwoOptMutation");
	    				 throw std::runtime_error( s );	
				}
	return true;				
}


int main()
{
  const unsigned POP_SIZE = 3, CHROM_SIZE = 8;
  unsigned i;

   // a chromosome randomizer
  eoInitPermutation <Chrom> random(CHROM_SIZE);
  
   // the population: 
  eoPop<Chrom> pop;
	
  // Evaluation
  eoEvalFuncPtr<Chrom> eval(  real_value );
 
  for (i = 0; i < POP_SIZE; ++i)
    {
      Chrom chrom(CHROM_SIZE);
      random(chrom);
      eval(chrom);
      pop.push_back(chrom);
    }
       
  // a twoOpt mutation
  eoTwoOptMutation <Chrom> twoOpt;
  
  for (i = 0; i < POP_SIZE; ++i)
    {
      std::cout << " Initial chromosome n°" << i << " : " << pop[i] << "..." <<  std::endl;
      twoOpt(pop[i]);
      std::cout << " ... becomes : " << pop[i] << " after twoOpt mutation" << std::endl;
      check_permutation(pop[i]);
    }
   
  return 0;
}

//-----------------------------------------------------------------------------
