#ifndef _doStatNormalMulti_h
#define _doStatNormalMulti_h

#include <boost/numeric/ublas/io.hpp>

#include "doStat.h"
#include "doNormalMulti.h"
#include "doDistrib.h"

template < typename EOT >
class doStatNormalMulti : public doStat< doNormalMulti< EOT > >
{
public:
    doStatNormalMulti( doNormalMulti< EOT >& distrib )
	: doStat< doNormalMulti< EOT > >( distrib )
    {}

    virtual void printOn(std::ostream& os) const
    {
	os << this->distrib().mean() << this->distrib().varcovar();
    }
};

#endif // !_doStatNormalMulti_h
