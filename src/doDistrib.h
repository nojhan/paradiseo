// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doDistrib_h
#define _doDistrib_h

#include <eoFunctor.h>

template < typename EOT >
class doDistrib : public eoFunctorBase
{
public:
    //! Alias for the type
    typedef EOT EOType;

    virtual ~doDistrib(){}
};

#endif // !_doDistrib_h
