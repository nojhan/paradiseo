//-----------------------------------------------------------------------------
// eoLottery.h
//-----------------------------------------------------------------------------

#ifndef eoLottery_h
#define eoLottery_h

//-----------------------------------------------------------------------------

#include <values.h>  // MINFLOAT
#include <numeric>  // accumulate
#include <eo>       // eoPop eoSelect

//-----------------------------------------------------------------------------
/// eoLottery: a selection method.
/// requires Chrom::Fitness to be float castable
//-----------------------------------------------------------------------------

template<class Chrom> class eoLottery: public eoSelect<Chrom>
{
 public:
  /// (Default) Constructor.
  eoLottery(const float& _rate = 1.0): rate(_rate) {}
  
  /// 
  void operator()(const eoPop<Chrom>& pop, eoPop<Chrom>& breeders) const
    {
      // scores of chromosomes
      vector<float> score(pop.size());

      // calculates accumulated scores for chromosomes
      for (unsigned i = 0; i < pop.size(); i++)
	score[i] = static_cast<float>(pop[i].fitness()); 

      float sum = accumulate(score.begin(), score.end(), MINFLOAT);
      transform(score.begin(), score.end(), score.begin(), 
		bind2nd(divides<float>(), sum));
      partial_sum(score.begin(), score.end(), score.begin());
      
      // generates random numbers
      vector<float> random(pop.size());
      generate(random.begin(), random.end(), eoUniform<float>(0,1));
      sort(random.begin(), random.end(), less<float>());
      
      // selection of chromosomes
      unsigned score_index = 0;   // position in score vector
      unsigned random_index = 0;  // position in random vector
      unsigned num_chroms = (unsigned)(rate * pop.size());
      do {
	if(random[random_index] < score[score_index]) 
	  {
	    breeders.push_back(pop[score_index]);
	    random_index++;
	  }
	else
	  if (score_index < pop.size())
	    score_index++;
	  else
	    fill_n(back_insert_iterator<eoPop<Chrom> >(breeders), 
		   num_chroms - breeders.size(), pop.back());
      } while (breeders.size() < num_chroms);
    }
  
 private:
  float rate;  // selection rate
};

//-----------------------------------------------------------------------------

#endif eoLottery_h
