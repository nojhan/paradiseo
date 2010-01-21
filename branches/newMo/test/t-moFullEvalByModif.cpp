#include "moTestClass.h"
#include <eval/moFullEvalByModif.h>

#include <cstdlib>
#include <cassert>

int main(){

	//Pas grand chose à faire: le gros du travail est fait par le voisin et l'eval

	std::cout << "[t-moFullEvalByModif] => START" << std::endl;

	Solution sol;
	moDummyBackableNeighbor neighbor;
	moDummyEval eval;

	//verif constructor
	moFullEvalByModif<moDummyBackableNeighbor> test(eval);

	sol.fitness(3);

	//verif operator()
	test(sol,neighbor);
	assert(sol.fitness()==3);

	std::cout << "[t-moFullEvalByModif] => OK" << std::endl;

	return EXIT_SUCCESS;
}
