//-----------------------------------------------------------------------------
// eoInclusion.h
//-----------------------------------------------------------------------------

#ifndef eoInclusion_h
#define eoInclusion_h

//-----------------------------------------------------------------------------

#include <iostream>

// EO includes
#include <eoPop.h>    
#include <eoMerge.h>

/*****************************************************************************
 * eoInclusion: A replacement algorithm.                                     *
 * Creates a new population by selecting the best individuals from the       *
 * breeders and original populations                                         *
 *****************************************************************************/

template<class Chrom> class eoInclusion: public eoMerge<Chrom>
{
 public:
  /// (Default) Constructor.
  eoInclusion(const float& _rate = 1.0): eoMerge<Chrom>( _rate ) {}

  /// Ctor from istream
  eoInclusion( istream& _is): eoBinPopOp<Chrom>( _is ) {};

  /// Dtor
  virtual ~eoInclusion() {};

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

  /** @name Methods from eoObject	*/
  //@{
  /** readFrom and printOn inherited from eoMerge */
  
  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  virtual string className() const {return "eoInclusion";};
  //@}
};

//-----------------------------------------------------------------------------

#endif eoInclusion_h
