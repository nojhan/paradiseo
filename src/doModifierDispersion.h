#ifndef _doModifierDispersion_h
#define _doModifierDispersion_h

#include <eoPop.h>
#include <eoFunctor.h>

#include "doModifier.h"

template < typename D >
class doModifierDispersion : public doModifier< D >, public eoBF< D&, eoPop< typename D::EOType >&, void >
{
public:
    // virtual void operator() ( D&, eoPop< D::EOType >& )=0 (provided by eoBF< A1, A2, R >)
};

#endif // !_doModifierDispersion_h
