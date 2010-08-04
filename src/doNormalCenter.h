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
	ublas::vector< AtomType > mean( distrib.size() );
	std::copy( mass.begin(), mass.end(), mean.begin() );
	distrib.mean() = mean;
    }
};

#endif // !_doNormalCenter_h
