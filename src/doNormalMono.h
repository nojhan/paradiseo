// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doNormalMono_h
#define _doNormalMono_h

#include "doDistrib.h"

template < typename EOT >
class doNormalMono : public doDistrib< EOT >
{
public:
    doNormalMono( const EOT& mean, const EOT& variance )
	: _mean(mean), _variance(variance)
    {
	assert(_mean.size() > 0);
	assert(_mean.size() == _variance.size());
    }

    unsigned int size()
    {
	assert(_mean.size() == _variance.size());
	return _mean.size();
    }

    EOT mean(){return _mean;}
    EOT variance(){return _variance;}

private:
    EOT _mean;
    EOT _variance;
};

#endif // !_doNormalMono_h
