//-----------------------------------------------------------------------------
// eoInsertion.h
//-----------------------------------------------------------------------------

#ifndef eoInsertion_h
#define eoInsertion_h

//-----------------------------------------------------------------------------

#include <eo>

/******************************************************************************
 * eoInsertion: A replacement algorithm.
 * Takes two populations: breeders and original populations. At the en of the 
 * process, the original population has chenge in the followin way:
 *            (1) the worst individuals haa been erased
 *            (2) the best individuals from the breeders has been added
 *****************************************************************************/

template<class Chrom> class eoInsertion: public eoMerge<Chrom>
{
 public:
  /// (Default) Constructor.
  eoInsertion(const float& _rate = 1.0): eoMerge(_rate) {}

  /**
   * Creates a new population based on breeders and original population
   * @param breeders The population of breeders.
   * @param pop The original population.
   */
  void operator()(const eoPop<Chrom>& breeders, eoPop<Chrom>& pop)
    {
      sort(pop.begin(), pop.end());
      
      if (rated() > 1)
	pop.erase(pop.end() + 
		  (int)(pop.size() * (rate() - 1) - breeders.size()),
		  pop.end());
      else
	{
	  cout << "eoInsertion no funciona con rate < 1"
	    exit(1);
	}
      
      copy(breeders.begin(), breeders.end(), 
	   back_insert_iterator<eoPop<Chrom> >(pop));
    }
};

//-----------------------------------------------------------------------------

#endif eoInsertion_h
