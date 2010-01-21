#include "moTestClass.h"

#include <cstdlib>
#include <cassert>

int main(){

	std::cout << "[t-moNeighbor] => START" << std::endl;

	moDummyNeighbor test1, test2;

	test1.fitness(3);
	test2=test1;

	assert(test1.fitness()==test2.fitness());

	moDummyNeighbor test3(test1);

	assert(test1.fitness()==test3.fitness());

	test1.printOn(std::cout);
	test2.printOn(std::cout);
	test3.printOn(std::cout);

	std::cout << "[t-moNeighbor] => OK" << std::endl;

	return EXIT_SUCCESS;
}
