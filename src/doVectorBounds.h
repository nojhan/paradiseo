#ifndef _doVectorBounds_h
#define _doVectorBounds_h

#include "doDistribParams.h"

template < typename EOT >
class doVectorBounds : public doDistribParams< EOT >
{
public:
    doVectorBounds(EOT _min, EOT _max)
	: doDistribParams< EOT >(2)
    {
	assert(_min.size() > 0);
	assert(_min.size() == _max.size());

	min() = _min;
	max() = _max;
    }

    doVectorBounds(const doVectorBounds& v)
	: doDistribParams< EOT >( v )
    {}

    EOT& min(){return this->param(0);}
    EOT& max(){return this->param(1);}
};

#endif // !_doVectorBounds_h
