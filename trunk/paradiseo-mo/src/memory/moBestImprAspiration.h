#ifndef _moBestImprAspiration_h
#define _moBestImprAspiration_h

#include <memory/moAspiration.h>

/**
 * Aspiration criteria accepts a solution better than the best so far
 */
template< class Neighbor >
class moBestImprAspiration : public moAspiration<Neighbor>
{
public:

	typedef typename Neighbor::EOT EOT;

	/**
	 * init the best solution
	 * @param _sol the best solution found
	 */
	void init(EOT & _sol){
		bestFoundSoFar = _sol;
	}

	/**
	 * update the "bestFoundSoFar" if a best solution is found
	 * @param _sol a solution
	 * @param _neighbor a neighbor
	 */
	void update(EOT & _sol, Neighbor & _neighbor){
		if (bestFoundSoFar.fitness() < _sol.fitness())
			bestFoundSoFar = _sol;
	}

	/**
	 * Test the tabu feature of the neighbor:
	 * test if the neighbor's fitness is better than the "bestFoundSoFar" fitness
	 * @param _sol a solution
	 * @param _neighbor a neighbor
	 * @return true if _neighbor fitness is better than the "bestFoundSoFar"
	 */
	bool operator()(EOT & _sol, Neighbor & _neighbor){
		return (bestFoundSoFar.fitness() < _neighbor.fitness());
	}

	/**
	 * Getter
	 * @return a reference on the best found so far solution
	 */
	EOT& getBest(){
		return bestFoundSoFar;
	}

private:
	EOT bestFoundSoFar;
};

#endif
