//-----------------------------------------------------------------------------
// eoBreeder.h
//-----------------------------------------------------------------------------

#ifndef eoGopBreeder_h
#define eoGopBreeder_h

//-----------------------------------------------------------------------------

/*****************************************************************************
 * eoBreeder: transforms a population using genetic operators.               *
 * For every operator there is a rated to be applyed.                        *
 *****************************************************************************/

#include <eoFunctor.h>
#include <eoPop.h>
#include <eoGOpSelector.h>
#include <eoIndiSelector.h>
#include <eoBackInserter.h>

/**
  Base class for breeders using generalized operators, I'm not sure if we
  will maintain the generalized operators in their current form, so
  it might change.
*/
template<class EOT> 
class eoGOpBreeder: public eoUF<eoPop<EOT>&, void>
{
 public:
  /// Default constructor.
  eoGOpBreeder( eoGOpSelector<EOT>& _opSel,
                    eoSelectOneIndiSelector<EOT>& _selector) 
                    : opSel( _opSel ), selector(_selector) 
        {}
  
  /**
   * Enlarges the population.
   * @param pop The population to be transformed.
   */
  void operator()(eoPop<EOT>& _pop) 
    {  
      unsigned size = _pop.size();
      
      for (unsigned i = 0; i < size; i++) 
	{ // and the one liner
	  opSel.selectOp()(selector.bind(_pop,size).bias(i), inserter.bind(_pop));
	}
    }
  
  /// The class name.
  string className() const { return "eoGOpBreeder"; }
  
 private:
  eoGOpSelector<EOT>&            opSel;
  eoSelectOneIndiSelector<EOT>&     selector;
  
  // the inserter can be local as there's no point in changing it from the outside
  eoBackInserter<EOT>     inserter;
};

#endif eoBreeder_h

