#include "moTestClass.h"

#include <cassert>

int main(){

	moDummyNeighbor n1, n3;

	n1.fitness(12);

	moDummyNeighbor n2(n1);
	assert(n1.fitness() == n2.fitness());

	n3=n1;
	assert(n1.fitness() == n3.fitness());

	return EXIT_SUCCESS;
}
