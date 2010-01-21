#include "moTestClass.h"
#include <eval/moFullEvalByCopy.h>

#include <cstdlib>
#include <cassert>

int main(){

	//Pas grand chose à faire: le gros du travail est fait par le voisin et l'eval

	std::cout << "[t-moFullEvalByCopy] => START" << std::endl;

	Solution sol;
	moDummyNeighbor neighbor;
	moDummyEval eval;

	//verif constructor
	moFullEvalByCopy<moDummyNeighbor> test(eval);

	sol.fitness(3);

	//verif operator()
	test(sol,neighbor);
	assert(sol.fitness()==3);

	std::cout << "[t-moFullEvalByCopy] => OK" << std::endl;

	return EXIT_SUCCESS;
}
