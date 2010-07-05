#ifndef _doBounderRng_h
#define _doBounderRng_h

#include "doBounder.h"

template < typename EOT >
class doBounderRng : public doBounder< EOT >
{
public:
    doBounderRng( EOT min, EOT max, eoRndGenerator< double > & rng )
	: doBounder< EOT >( min, max ), _rng(rng)
    {}

    void operator()( EOT& x )
    {
	unsigned int size = x.size();
	assert(size > 0);

	for (unsigned int d = 0; d < size; ++d) // browse all dimensions
	    {

		// FIXME: attention: les bornes RNG ont les memes bornes quelque soit les dimensions idealement on voudrait avoir des bornes differentes pour chaque dimensions.

		if (x[d] < this->min()[d] || x[d] > this->max()[d])
		    {
			x[d] = _rng();
		    }
	    }
    }

private:
    eoRndGenerator< double> & _rng;
};

#endif // !_doBounderRng_h
