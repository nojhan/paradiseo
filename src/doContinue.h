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

#endif // !_doContinue_h
