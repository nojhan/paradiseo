#ifndef _doStatNormalMono_h
#define _doStatNormalMono_h

#include "doStat.h"
#include "doNormalMono.h"

template < typename EOT >
class doStatNormalMono : public doStat< doNormalMono< EOT > >
{
public:
    doStatNormalMono( doNormalMono< EOT >& distrib )
	: doStat< doNormalMono< EOT > >( distrib )
    {}

    virtual void printOn(std::ostream& os) const
    {
	os << this->distrib().mean() << " " << this->distrib().variance();
    }
};

#endif // !_doStatNormalMono_h
