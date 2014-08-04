#ifndef _Rosenbrock_h
#define _Rosenbrock_h

#include <paradiseo/eo.h>
#include <paradiseo/eo/es.h>
#include <paradiseo/eo/es/eoRealInitBounded.h>
#include <paradiseo/eo/es/eoRealOp.h>
#include <paradiseo/eo/es/eoEsChromInit.h>
#include <paradiseo/eo/es/eoRealOp.h>
#include <paradiseo/eo/es/make_real.h>
#include <paradiseo/eo/apply.h>
#include <paradiseo/eo/eoProportionalCombinedOp.h>

template < typename EOT >
class Rosenbrock : public eoEvalFunc< EOT >
{
public:
    typedef typename EOT::AtomType AtomType;

    virtual void operator()( EOT& p )
    {
	if (!p.invalid())
	    return;

	p.fitness( _evaluate( p ) );
    }

private:
    AtomType _evaluate( EOT& p )
    {
	AtomType r = 0.0;

	for (unsigned int i = 0; i < p.size() - 1; ++i)
	    {
		r += p[i] * p[i];
	    }

	return r;
    }
};

#endif // !_Rosenbrock_h
