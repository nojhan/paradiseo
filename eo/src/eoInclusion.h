//-----------------------------------------------------------------------------
// eoInclusion.h
//-----------------------------------------------------------------------------

#ifndef eoInclusion_h
#define eoInclusion_h

//-----------------------------------------------------------------------------

#include <eoPop.h>    // eoPop 
#include <eoPopOps.h> // eoMerge

/*****************************************************************************
 * eoInclusion: A replacement algorithm.                                     *
 * Creates a new population by selecting the best individuals from the       *
 * breeders and original populations                                         *
 *****************************************************************************/

template<class Chrom> class eoInclusion: public eoMerge<Chrom>
{
 public:
  /// (Default) Constructor.
  eoInclusion(const float& _rate = 1.0): eoMerge<Chrom>(_rate) {}

  /**
   * Creates a new population based on breeders and original populations.
   * @param breeders The population of breeders.
   * @param pop The original population.
   */
  void operator()(eoPop<Chrom>& breeders, eoPop<Chrom>& pop)
    {
      unsigned target = min(static_cast<unsigned>(rint(pop.size() * rate())), 
			    pop.size() + breeders.size());
      
      copy(breeders.begin(), breeders.end(), 
	   back_insert_iterator<eoPop<Chrom> >(pop));
      partial_sort(pop.begin(), pop.begin() + target, pop.end(),
		   greater<Chrom>());
      pop.erase(pop.begin() + target, pop.end());
    }
};

//-----------------------------------------------------------------------------

#endif eoInclusion_h
