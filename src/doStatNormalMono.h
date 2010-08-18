// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doStatNormalMono_h
#define _doStatNormalMono_h

#include "doStat.h"
#include "doNormalMono.h"

template < typename EOT >
class doStatNormalMono : public doDistribStat< doNormalMono< EOT > >
{
public:
    using doDistribStat< doNormalMono< EOT > >::value;

    doStatNormalMono( std::string desc = "" )
	: doDistribStat< doNormalMono< EOT > >( desc )
    {}

    void operator()( const doNormalMono< EOT >& distrib )
    {
	value() = "\n# ====== mono normal distribution dump =====\n";

	std::ostringstream os;
	os << distrib.mean() << " " << distrib.variance() << std::endl;

	value() += os.str();
    }
};

#endif // !_doStatNormalMono_h
