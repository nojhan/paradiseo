// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doStatUniform_h
#define _doStatUniform_h

#include "doStat.h"
#include "doUniform.h"

template < typename EOT >
class doStatUniform : public doDistribStat< doUniform< EOT > >
{
public:
    using doDistribStat< doUniform< EOT > >::value;

    doStatUniform( std::string desc = "" )
	: doDistribStat< doUniform< EOT > >( desc )
    {}

    void operator()( const doUniform< EOT >& distrib )
    {
	value() = "\n# ====== uniform distribution dump =====\n";

	std::ostringstream os;
	os << distrib.min() << " " << distrib.max() << std::endl;

	value() += os.str();
    }
};

#endif // !_doStatUniform_h
