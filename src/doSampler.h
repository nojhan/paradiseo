// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doSampler_h
#define _doSampler_h

#include <eoFunctor.h>

#include "doBounder.h"

template < typename D >
class doSampler : public eoUF< D&, typename D::EOType >
{
public:
    typedef typename D::EOType EOType;

    doSampler(doBounder< EOType > & bounder)
	: _bounder(bounder)
    {}

    // virtual EOType operator()( D& ) = 0 (provided by eoUF< A1, R >)

    virtual EOType sample( D& ) = 0;

    EOType operator()( D& distrib )
    {
	unsigned int size = distrib.size();
	assert(size > 0);


	//-------------------------------------------------------------
	// Point we want to sample to get higher a set of points
	// (coordinates in n dimension)
	// x = {x1, x2, ..., xn}
	// the sample method is implemented in the derivated class
	//-------------------------------------------------------------

	EOType solution(sample(distrib));

	//-------------------------------------------------------------


	//-------------------------------------------------------------
	// Now we are bounding the distribution thanks to min and max
	// parameters.
	//-------------------------------------------------------------

	_bounder(solution);

	//-------------------------------------------------------------


	return solution;
    }

private:
    //! Bounder functor
    doBounder< EOType > & _bounder;
};

#endif // !_doSampler_h
