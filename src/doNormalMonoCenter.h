// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doNormalMonoCenter_h
#define _doNormalMonoCenter_h

#include "doModifierMass.h"
#include "doNormalMono.h"

template < typename EOT >
class doNormalMonoCenter : public doModifierMass< doNormalMono< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    void operator() ( doNormalMono< EOT >& distrib, EOT& mass )
    {
	distrib.mean() = mass;
    }
};

#endif // !_doNormalMonoCenter_h
