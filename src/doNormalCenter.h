#ifndef _doNormalCenter_h
#define _doNormalCenter_h

#include "doModifierMass.h"
#include "doNormal.h"

template < typename EOT >
class doNormalCenter : public doModifierMass< doNormal< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    void operator() ( doNormal< EOT >& distrib, EOT& mass )
    {
	distrib.mean() = mass; // vive les references!!!
    }
};

#endif // !_doNormalCenter_h
