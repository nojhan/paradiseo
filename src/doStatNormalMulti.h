#ifndef _doStatNormalMulti_h
#define _doStatNormalMulti_h

#include <boost/numeric/ublas/io.hpp>

#include "doStat.h"
#include "doNormalMulti.h"

template < typename EOT >
class doStatNormalMulti : public doDistribStat< doNormalMulti< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    using doDistribStat< doNormalMulti< EOT > >::value;

    doStatNormalMulti( std::string desc = "" )
	: doDistribStat< doNormalMulti< EOT > >( desc )
    {}

    void operator()( const doNormalMulti< EOT >& distrib )
    {
	value() = "\n# ====== multi normal distribution dump =====\n";

	std::ostringstream os;

	os << distrib.mean() << " " << distrib.varcovar() << std::endl;

	// ublas::vector< AtomType > mean = distrib.mean();
	// std::copy(mean.begin(), mean.end(), std::ostream_iterator< std::string >( os, " " ));

	// ublas::symmetric_matrix< AtomType, ublas::lower > varcovar = distrib.varcovar();
	// std::copy(varcovar.begin(), varcovar.end(), std::ostream_iterator< std::string >( os, " " ));

	// os << std::endl;

	value() += os.str();
    }
};

#endif // !_doStatNormalMulti_h
