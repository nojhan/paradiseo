// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoLottery.h
//   Implements the lottery procedure for selection
// (c) GeNeura Team, 1998
/* 
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */
//-----------------------------------------------------------------------------

#ifndef eoLottery_h
#define eoLottery_h

//-----------------------------------------------------------------------------

#include <functional>  // 
#include <numeric>     // accumulate
#include <eo>          // eoPop eoSelect MINFLOAT

//-----------------------------------------------------------------------------
/// eoLottery: a selection method.
/// requires Chrom::Fitness to be float castable
//-----------------------------------------------------------------------------

template<class Chrom> class eoLottery: public eoBinPopOp<Chrom>
{
 public:
  /// (Default) Constructor.
  eoLottery(const float& _rate = 1.0): rate(_rate) {}
  
  /// 
  void operator()( eoPop<Chrom>& pop, eoPop<Chrom>& breeders) 
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
      vector<float> random(rint(rate * pop.size()));
      generate(random.begin(), random.end(), eoUniform<float>(0,1));
      sort(random.begin(), random.end(), less<float>());
      
      // selection of chromosomes
      unsigned score_index = 0;   // position in score vector
      unsigned random_index = 0;  // position in random vector
      while (breeders.size() < random.size()) 
	{
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
		     random.size() - breeders.size(), pop.back());
	}
    }
  
 private:
  float rate;  // selection rate
};

//-----------------------------------------------------------------------------

#endif eoLottery_h
