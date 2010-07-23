#ifndef _doNormalParams_h
#define _doNormalParams_h

#include "doDistribParams.h"

template < typename EOT >
class doNormalParams : public doDistribParams< EOT >
{
public:
    doNormalParams(EOT _mean, EOT _variance)
	: doDistribParams< EOT >(2)
    {
	assert(_mean.size() > 0);
	assert(_mean.size() == _variance.size());

	mean() = _mean;
	variance() = _variance;
    }

    doNormalParams(const doNormalParams& p)
	: doDistribParams< EOT >( p )
    {}

    EOT& mean(){return this->param(0);}
    EOT& variance(){return this->param(1);}
};

#endif // !_doNormalParams_h
