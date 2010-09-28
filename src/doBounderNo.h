// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doBounderNo_h
#define _doBounderNo_h

#include "doBounder.h"

template < typename EOT >
class doBounderNo : public doBounder< EOT >
{
public:
    void operator()( EOT& ) {}
};

#endif // !_doBounderNo_h
