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

template<class Chrom> class eoGeneration: public eoTransform<Chrom>
{
 public:
  /// Constructor.
  eoGeneration(eoSelect& _select, 
	       eoTranform& _transform, 
	       eoMerge& _replace):
    eoTransform<Chrom>() {}
  
  /**
   *
   */
  void operator()(eoPop& pop)
    {
      eoPop breeders;
      
      select(pop, breeders);
      transform(breeders);
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
