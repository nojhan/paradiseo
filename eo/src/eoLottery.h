//-----------------------------------------------------------------------------
// eoLottery.h
//-----------------------------------------------------------------------------

#ifndef eoLottery_h
#define eoLottery_h

//-----------------------------------------------------------------------------

#include <eo>

//-----------------------------------------------------------------------------
// eoLottery
//-----------------------------------------------------------------------------

template<class Chrom> class eoLottery: public eoSelect<Chrom>
{
 public:
  eoLottery(const float& rate = 1.0): eoLottery(rate) {}
  
  void operator()(const eoPop<Chrom>& pop, eoPop<Chrom>& breeders)
    {
      // scores of chromosomes
      vector<float> score(pop.size());
      
      // calculates accumulated scores for chromosomes
      transform(pop.begin(), pop.end(), score.begin(), fitness);
      float sum = accumulate(score.begin(), score.end(), MINFLOAT);
      transform(score.begin(), score.end(), score.begin(), 
		bind2nd(divides<float>(), sum));
      partial_sum(score.begin(), score.end(), score.begin());
      
      // generates random numbers
      vector<float> random(pop.size());
      generate(random.begin(), random.end(), Uniform<float>(0,1));
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
	    fill_n(back_insert_iterator<Pop>(breeders), 
		   num_chroms - breeders.size(), pop.back());
      } while (breeders.size() < num_chroms);
    }
};

//-----------------------------------------------------------------------------

#endif eoLottery_h
