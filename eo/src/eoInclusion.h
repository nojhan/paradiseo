//-----------------------------------------------------------------------------
// eoInclusion.h
//-----------------------------------------------------------------------------

#ifndef eoInclusion_h
#define eoInclusion_h

//-----------------------------------------------------------------------------

#include <eo>

/******************************************************************************
 * eoInclusion: A replacement algorithm.
 * Creates a new population by selecting the best individuals from the 
 * breeders and original populations
 *****************************************************************************/

template<class Chrom> class eoInclusion: public eoMerge<Chrom>
{
 public:
  /// (Default) Constructor.
  eoInclusion(const float& _rate = 1.0): eoMerge(_rate) {}

  /**
   * Creates a new population based on breeders and original populations.
   * @param breeders The population of breeders.
   * @param pop The original population.
   */
  void operator()(const eoPop<Chrom>& breeders, eoPop<Chrom>& pop)
    {
      eoPop<Chrom> all, tmp = breeders;
      
      sort(tmp.begin(), tmp.end());
      sort(pop.begin(), pop.end());
      
      merge(tmp.begin(), tmp.end(),
	    pop.begin(), pop.end(),
	    back_insert_iterator<eoPop<Chrom> >(all));
      
      all.erase(all.begin(),
		all.begin() + (unsigned)(all.size() - pop.size() * rate()));
      
      pop.swap(all);
    }
};

//-----------------------------------------------------------------------------

#endif eoInclusion_h
