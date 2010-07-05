#ifndef _Rosenbrock_h
#define _Rosenbrock_h

#include <eo>
#include <es.h>
#include <es/eoRealInitBounded.h>
#include <es/eoRealOp.h>
#include <es/eoEsChromInit.h>
#include <es/eoRealOp.h>
#include <es/make_real.h>
#include <apply.h>
#include <eoProportionalCombinedOp.h>

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
