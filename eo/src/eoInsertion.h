//-----------------------------------------------------------------------------
// eoInsertion.h
//-----------------------------------------------------------------------------

#ifndef eoInsertion_h
#define eoInsertion_h

//-----------------------------------------------------------------------------

#include <eo>

//-----------------------------------------------------------------------------
// eoInsertion
//-----------------------------------------------------------------------------

template<class Chrom> class eoInsertion: public eoMerge<Chrom>
{
 public:
  /// (Default) Constructor.
  eoInsertion(const float& _rate = 1.0): eoMerge(_rate) {}

  ///
  /// @param breeders
  /// @param pop
  void operator()(const eoPop<Chrom>& breeders, eoPop<Chrom>& pop)
    {
      cout << pop << endl;
      
      sort(pop.begin(), pop.end());
      
      cout << pop << endl;

      cout << "cte = " 
	   << (int)(pop.size() * (rate() - 1) - breeders.size()) << endl;

      pop.erase(pop.end() + (int)(pop.size() * (rate() - 1) - breeders.size()),
		pop.end());

      cout << "cte = " 
	   << (int)(pop.size() * (rate() - 1) - breeders.size()) << endl;

      copy(breeders.begin(), breeders.end(), 
	   back_insert_iterator<eoPop<Chrom> >(pop));
    }
};

//-----------------------------------------------------------------------------

#endif eoInsertion_h
