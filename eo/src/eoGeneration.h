//-----------------------------------------------------------------------------
// eoGeneration.h
//-----------------------------------------------------------------------------

#ifndef eoGeneration_h
#define eoGeneration_h

//-----------------------------------------------------------------------------

#include <eoPop.h>     // eoPop
#include <eoPopOps.h>  // eoSelect, eoTranform, eoMerge

//-----------------------------------------------------------------------------
// eoGeneration
//-----------------------------------------------------------------------------

template<class Chrom> class eoGeneration
{
 public:
  /// Constructor.
  eoGeneration(eoSelect&   _select, 
	       eoTranform& _transform, 
	       eoMerge&    _replace):
    select(_select), transform(_transform), replace(_replace) {}
  
  /// apply one generation of evolution to the population
  void operator()(eoPop& pop)
    {
      eoPop breeders;
      
      select(pop, breeders);
      transform(breeders);
      for_each(pop.begin(), pop.end(), Chrom::Fitness);
      replace(breeders, pop);
    }
  
  /// Class name.
  string className() const { return "eoGeneration"; }
  
 private:
  eoSelect&   select;
  eoTranform& transform;
  eoMerge&    replace;
};

//-----------------------------------------------------------------------------

#endif eoGeneration_h
