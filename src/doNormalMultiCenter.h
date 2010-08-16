#ifndef _doNormalMultiCenter_h
#define _doNormalMultiCenter_h

#include "doModifierMass.h"
#include "doNormalMulti.h"

template < typename EOT >
class doNormalMultiCenter : public doModifierMass< doNormalMulti< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    void operator() ( doNormalMulti< EOT >& distrib, EOT& mass )
    {
	ublas::vector< AtomType > mean( distrib.size() );
	std::copy( mass.begin(), mass.end(), mean.begin() );
	distrib.mean() = mean;
    }
};

#endif // !_doNormalMultiCenter_h
