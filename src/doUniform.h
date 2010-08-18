// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doUniform_h
#define _doUniform_h

#include "doDistrib.h"
#include "doVectorBounds.h"

template < typename EOT >
class doUniform : public doDistrib< EOT >, public doVectorBounds< EOT >
{
public:
    doUniform(EOT min, EOT max)
	: doVectorBounds< EOT >(min, max)
    {}
};

#endif // !_doUniform_h
