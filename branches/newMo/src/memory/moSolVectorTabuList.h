#ifndef _moSolVectorTabuList_h
#define _moSolVectorTabuList_h

#include <memory/moTabuList.h>

/**
 * Tabu List of solution stocked in a vector
 */
template<class Neighbor >
class moSolVectorTabuList : public moTabuList<Neighbor>
{
public:
	typedef typename Neighbor::EOT EOT;

	/**
	 * Constructor
	 * @param _maxSize maximum size of the tabu list
	 */
	moSolVectorTabuList(unsigned int _maxSize) : maxSize(_maxSize){
		tabuList.reserve(_maxSize);
		tabuList.resize(0);
	}

	/**
	 * init the tabuList by clearing the memory
	 * @param _sol the current solution
	 */
	void init(EOT & _sol){
		clearMemory();
	}


	/**
	 * add a new solution in the tabuList
	 * @param _sol the current solution
	 * @param _neighbor the current neighbor (unused)
	 */
	void add(EOT & _sol, Neighbor & _neighbor)
	{
		if(tabuList.size() < maxSize)
			tabuList.push_back(_sol);
		else{
			tabuList[index%maxSize] = _sol;
			index++;
		}
	}

	/**
	 * update the tabulist
	 * @param _sol the current solution
	 * @param _neighbor the current neighbor (unused)
	 */
	void update(EOT & _sol, Neighbor & _neighbor){
		// ???
	}

	/**
	 * check if the solution is tabu
	 * @param _sol the current solution
	 * @param _neighbor the current neighbor (unused)
	 * @return true if tabuList contains _sol
	 */
	bool check(EOT & _sol, Neighbor & _neighbor){
		for(unsigned int i=0; i<tabuList.size(); i++){
			if (tabuList[i] == _sol)
				return true;
		}
		return false;
	}

	/**
	 * clearMemory: remove all solution of the tabuList
	 */
	virtual void clearMemory(){
		tabuList.resize(0);
		index = 0;
	}


private:

	std::vector<EOT> tabuList;
	unsigned int maxSize;
	unsigned long index;

};

#endif
