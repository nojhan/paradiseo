//-----------------------------------------------------------------------------
// eoGA.h
//-----------------------------------------------------------------------------

#ifndef eoGA_h
#define eoGA_h

//-----------------------------------------------------------------------------

#include <eoPop.h>     // eoPop
#include <eoPopOps.h>  // eoSelect, eoTranform, eoMerge

//-----------------------------------------------------------------------------
// eoGA
//-----------------------------------------------------------------------------

class eoGA
{
 public:
  /// Constructor.
  eoGA(eoSelect& _select, eoTranform& _transform, eoMerge& _replace)
    {
    }
  
  /// 
  void operator()(eoPop& pop)
    {
      eoPop breeders;
      
      select(pop, breeders);
      transform(breeders);
      replace(breeders, pop);
    }
  
 private:
  eoSelect&   select;
  eoTranform& transform;
  eoMerge&    replace;
};

//-----------------------------------------------------------------------------

#endif eoGA_h
