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

#include "eoPopOps.h"
#include "eoGOpSelector.h"
#include "eoIndiSelector.h"
#include "eoBackInserter.h"

template<class EOT> 
class eoGOpBreeder: public eoMonPopOp<EOT>
{
 public:
  /// Default constructor.
  eoGOpBreeder( eoGOpSelector<EOT>& _opSel,
                    eoPopIndiSelector<EOT>& _selector) 
                    : opSel( _opSel ), selector(_selector) 
        {}
  
  /// Destructor.
  virtual ~eoGOpBreeder() {}

  /**
   * Enlarges the population.
   * @param pop The population to be transformed.
   */
  void operator()(eoPop<EOT>& _pop) 
  {  
	  int size = _pop.size();

      for (unsigned i = 0; i < size; i++) 
      { // and the one liner
		opSel.selectOp()(selector(_pop,size, i), inserter(_pop));
      }
	}
  
  /// The class name.
  string className() const { return "eoGOpBreeder"; }
  
 private:
  eoGOpSelector<EOT>&            opSel;
  eoPopIndiSelector<EOT>&     selector;
  
  // the inserter can be local as there's no point in changing it from the outside
  eoBackInserter<EOT>     inserter;
};

#endif eoBreeder_h

