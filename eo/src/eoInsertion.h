//-----------------------------------------------------------------------------
// eoInsertion.h
//-----------------------------------------------------------------------------

#ifndef eoInsertion_h
#define eoInsertion_h

//-----------------------------------------------------------------------------

#include <eo>

/******************************************************************************
 * eoInsertion: A replacement algorithm.
 * Creates a new population with all the breeders and the best individuals 
 * from the original population.
 *****************************************************************************/

template<class Chrom> class eoInsertion: public eoMerge<Chrom>
{
 public:
  /// (Default) Constructor.
  eoInsertion(const float& _rate = 1.0): eoMerge<Chrom>(_rate) {}

  /**
   * Creates a new population based on breeders and original populations.
   * @param breeders The population of breeders.
   * @param pop The original population.
   */
  void operator()(const eoPop<Chrom>& breeders, eoPop<Chrom>& pop)
    {
      int new_size = static_cast<int>(pop.size() * rate());
      
      if (new_size == breeders.size())
	{
	  pop = breeders;
	}
      else if (new_size < breeders.size())
	{
	  pop = breeders;
	  sort(pop.begin(), pop.end());
	  pop.erase(pop.begin(), pop.begin() - new_size + pop.size());
	}
      else
	{
	  sort(pop.begin(), pop.end());
	  pop.erase(pop.begin(), 
		    pop.begin() + breeders.size() + pop.size() - new_size);
	  copy(breeders.begin(), breeders.end(), 
	       back_insert_iterator<eoPop<Chrom> >(pop));
	}
    }
};

//-----------------------------------------------------------------------------

#endif eoInsertion_h
