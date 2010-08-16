#ifndef _doStatUniform_h
#define _doStatUniform_h

#include "doStat.h"
#include "doUniform.h"

template < typename EOT >
class doStatUniform : public doStat< doUniform< EOT > >
{
public:
    doStatUniform( doUniform< EOT >& distrib )
	: doStat< doUniform< EOT > >( distrib )
    {}

    virtual void printOn(std::ostream& os) const
    {
	os << this->distrib().min() << " " << this->distrib().max();
    }
};

#endif // !_doStatUniform_h
