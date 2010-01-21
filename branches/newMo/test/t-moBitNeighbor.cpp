#include <neighborhood/moBitNeighbor.h>

#include <cstdlib>
#include <cassert>

int main(){

	std::cout << "[t-moBitNeighbor] => START" << std::endl;

	//init sol
	eoBit<int> sol;
	sol.push_back(true);
	sol.push_back(false);
	sol.push_back(true);

	//verif du constructeur vide
	moBitNeighbor<int> test1;
	assert(test1.index()==0);

	//verif du constructeur indiquant le bit
	moBitNeighbor<int> hop(34);
	assert(hop.index()==34);

	//verif du setter d'index et du constructeur de copy
	test1.index(6);
	test1.fitness(2);
	moBitNeighbor<int> test2(test1);
	assert(test2.index()==6);
	assert(test2.fitness()==2);

	//verif du getter
	assert(test1.index()==6);

	//verif de l'operateur=
	test1.fitness(8);
	test1.index(2);
	test2=test1;
	assert(test2.fitness()==8);
	assert(test2.index()==2);

	//verif de move
	test2.move(sol);
	assert(!sol[2]);

	//verif de moveBack
	test2.moveBack(sol);
	assert(sol[2]);

	test1.printOn(std::cout);
	test2.printOn(std::cout);

	std::cout << "[t-moBitNeighbor] => OK" << std::endl;
	return EXIT_SUCCESS;
}
