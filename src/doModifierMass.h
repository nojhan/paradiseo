// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doModifierMass_h
#define _doModifierMass_h

#include <eoFunctor.h>

#include "doModifier.h"

template < typename D >
class doModifierMass : public doModifier< D >, public eoBF< D&, typename D::EOType&, void >
{
public:
    //typedef typename D::EOType::AtomType AtomType; // does not work !!!

    // virtual void operator() ( D&, D::EOType& )=0 (provided by eoBF< A1, A2, R >)
};

#endif // !_doModifierMass_h
