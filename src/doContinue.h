// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doContinue_h
#define _doContinue_h

#include <eoFunctor.h>
#include <eoPop.h>
#include <eoPersistent.h>

//! eoContinue< EOT > classe fitted to Distribution Object library

template < typename D >
class doContinue : public eoUF< const D&, bool >, public eoPersistent
{
public:
    virtual std::string className(void) const { return "doContinue"; }

    void readFrom(std::istream&)
    {
	/* It should be implemented by subclasses ! */
    }

    void printOn(std::ostream&) const
    {
	/* It should be implemented by subclasses ! */
    }
};

template < typename D >
class doDummyContinue : public doContinue< D >
{
    bool operator()(const D&){ return true; }

    virtual std::string className() const { return "doNoContinue"; }
};

#endif // !_doContinue_h
