// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doUniformCenter_h
#define _doUniformCenter_h

#include "doModifierMass.h"
#include "doUniform.h"

template < typename EOT >
class doUniformCenter : public doModifierMass< doUniform< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    void operator() ( doUniform< EOT >& distrib, EOT& mass )
    {
	for (unsigned int i = 0, n = mass.size(); i < n; ++i)
	    {
		AtomType& min = distrib.min()[i];
		AtomType& max = distrib.max()[i];

		AtomType range = (max - min) / 2;

		min = mass[i] - range;
		max = mass[i] + range;
	    }
    }
};

#endif // !_doUniformCenter_h
