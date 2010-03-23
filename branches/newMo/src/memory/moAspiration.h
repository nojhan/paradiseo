#ifndef _moAspiration_h
#define _moAspiration_h

#include <eoFunctor.h>

/**
 * Abstract class for Aspiration Criteria
 */
template< class Neighbor >
class moAspiration : public eoBF<typename Neighbor::EOT &, Neighbor &, bool>
{
public:
	typedef typename Neighbor::EOT EOT;

	virtual void init(EOT & _sol) = 0;
	virtual void update(EOT & _sol, Neighbor & _neighbor) = 0;
};

#endif
