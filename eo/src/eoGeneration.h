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
  eoGeneration(eoSelect<Chrom>&    _select, 
	       eoTransform<Chrom>& _transform, 
	       eoMerge<Chrom>&     _replace):
    select(_select), transform(_transform), replace(_replace) {}
  
  /// Apply one generation of evolution to the population.
  template<class Evaluator> void operator()(eoPop<Chrom>& pop, 
					    Evaluator evaluator)
    {
      eoPop<Chrom> breeders;
      
      select(pop, breeders);
      transform(breeders);
      for_each(breeders.begin(), breeders.end(), evaluator);
      replace(breeders, pop);
    }
  
  /// Class name.
  string className() const { return "eoGeneration"; }
  
 private:
  eoSelect<Chrom>&    select;
  eoTransform<Chrom>& transform;
  eoMerge<Chrom>&     replace;
};

//-----------------------------------------------------------------------------

#endif eoGeneration_h
