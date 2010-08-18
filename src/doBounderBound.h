// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doBounderBound_h
#define _doBounderBound_h

#include "doBounder.h"

template < typename EOT >
class doBounderBound : public doBounder< EOT >
{
public:
    doBounderBound( EOT min, EOT max )
	: doBounder< EOT >( min, max )
    {}

    void operator()( EOT& x )
    {
	unsigned int size = x.size();
	assert(size > 0);

	for (unsigned int d = 0; d < size; ++d) // browse all dimensions
	    {
		if (x[d] < this->min()[d])
		    {
			x[d] = this->min()[d];
			continue;
		    }

		if (x[d] > this->max()[d])
		    {
			x[d] = this->max()[d];
		    }
	    }
    }
};

#endif // !_doBounderBound_h
