#ifndef _doNormal_h
#define _doNormal_h

#include "doDistrib.h"
#include "doNormalParams.h"

template < typename EOT >
class doNormal : public doDistrib< EOT >, public doNormalParams< EOT >
{
public:
    typedef typename EOT::AtomType AtomType;

    doNormal( const EOT& mean, const ublas::symmetric_matrix< AtomType, ublas::lower >& varcovar )
	: doNormalParams< EOT >( mean, varcovar )
    {}
};

#endif // !_doNormal_h
