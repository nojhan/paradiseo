#ifndef _moTabuList_h
#define _moTabuList_h

#include <memory/moMemory.h>

/**
 * Abstract class for the Tabu List
 */
template< class Neighbor >
class moTabuList : public moMemory<Neighbor>
{
public:
	typedef typename Neighbor::EOT EOT;

	/**
	 * Check if a neighbor is tabu or not
	 * @param _sol the current solution
	 * @param _neighbor the neighbor
	 * @return true if the neighbor is tabu
	 */
	virtual bool check(EOT & _sol, Neighbor & _neighbor) = 0;
};

#endif
