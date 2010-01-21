#include <neighborhood/moBitNeighborhood.h>

#include <cstdlib>
#include <cassert>

int main(){

	std::cout << "[t-moBitNeighborhood] => START" << std::endl;

	//init sol
	eoBit<int> sol;
	sol.push_back(true);
	sol.push_back(false);
	sol.push_back(true);

	moBitNeighbor<int> neighbor;

	//verif du constructeur vide
	moBitNeighborhood<moBitNeighbor<int> > test;
	assert(test.position()==0);

	//verif du hasneighbor
	assert(test.hasNeighbor(sol));

	//verif de init
	test.init(sol, neighbor);
	assert(neighbor.index()==0);
	assert(test.position()==0);

	//verif du next
	test.next(sol, neighbor);
	assert(neighbor.index()==1);
	assert(test.position()==1);

	//verif du cont
	test.next(sol, neighbor);
	assert(test.cont(sol));
	test.next(sol, neighbor);
	assert(!test.cont(sol));

	std::cout << "[t-moBitNeighborhood] => OK" << std::endl;
	return EXIT_SUCCESS;
}
