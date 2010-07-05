#ifndef _doNormal_h
#define _doNormal_h

#include "doDistrib.h"
#include "doNormalParams.h"

template < typename EOT >
class doNormal : public doDistrib< EOT >, public doNormalParams< EOT >
{
public:
    doNormal(EOT mean, EOT variance)
	: doNormalParams< EOT >(mean, variance)
    {}
};

#endif // !_doNormal_h
