/*
 <t-moGPUXChange.cu>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

 Karima Boufaras, Thé Van LUONG

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

 As a counterpart to the access to the source code and  rights to copy,
 modify and redistribute granted by the license, users are provided only
 with a limited warranty  and the software's author,  the holder of the
 economic rights,  and the successive licensors  have only  limited liability.

 In this respect, the user's attention is drawn to the risks associated
 with loading,  using,  modifying and/or developing or reproducing the
 software by the user in light of its specific status of free software,
 that may mean  that it is complicated to manipulate,  and  that  also
 therefore means  that it is reserved for developers  and  experienced
 professionals having in-depth computer knowledge. Users are therefore
 encouraged to load and test the software's suitability as regards their
 requirements in conditions enabling the security of their systems and/or
 data to be ensured and,  more generally, to use and operate it in the
 same conditions as regards security.
 The fact that you are presently reading this means that you have had
 knowledge of the CeCILL license and that you accept its terms.

 ParadisEO WebSite : http://paradiseo.gforge.inria.fr
 Contact: paradiseo-help@lists.gforge.inria.fr
 */

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <eoInt.h>
#include <neighborhood/moGPUXSwapN.h>
#include <neighborhood/moGPUXChange.h>
#include <neighborhood/moGPUNeighborhoodSizeUtils.h>
#include <eo>

typedef eoInt<eoMaximizingFitness> Solution;
typedef moGPUXSwapN<Solution> Neighbor;
typedef moGPUXChange<Neighbor> Neighborhood;

int main() {

	std::cout << "[t-moGPUXChange] => START" << std::endl;

	//test factorial
	assert(factorial(10) == 3628800);

	//test sizeMapping
	assert(sizeMapping(6, 1) == 6);
	assert(sizeMapping(6, 2) == 15);
	assert(sizeMapping(6, 3) == 20);
	assert(sizeMapping(6, 4) == 15);
	assert(sizeMapping(6, 5) == 6);

	Solution sol(6);
	Neighbor neighbor(2);
	Neighbor neighbor1;

	unsigned int * first;
	unsigned int * second;
	unsigned int * third;
	unsigned int * forth;

	first = new unsigned int[15];
	second = new unsigned int[15];
	third = new unsigned int[15];
	forth = new unsigned int[15];

	unsigned int id = 0;
	for (unsigned int i = 0; i < 5; i++)
		for (unsigned int j = i + 1; j < 6; j++) {
			first[id] = i;
			second[id] = j;
			id++;
		}

	//test Constructor
	Neighborhood neighborhood(15, 2);
	//test x-change getter
	assert(neighborhood.getXChange() == 2);
	//test neighborhoodSize getter
	assert(neighborhood.getNeighborhoodSize() == 15);
	//test current index getter
	assert(neighborhood.position() == 0);

	//test neighborhood methods

	//test hasNeighbor
	assert(neighborhood.hasNeighbor(sol) == true);

	//test init
	neighborhood.init(sol, neighbor);
	assert(neighbor.index() == 0);
	assert(neighbor.getXChange() == 2);
	assert(neighborhood.position() == 0);

	//test Mapping
	assert(neighbor.getIndice(0) == first[neighborhood.position()]);
	assert(neighbor.getIndice(1) == second[neighborhood.position()]);

	//test next & cont
	unsigned i = 1;
	while (neighborhood.cont(sol)) {
		neighborhood.next(sol, neighbor);
		assert(neighborhood.position() == i);
		//test Mapping
		assert(neighbor.getIndice(0) == first[neighborhood.position()]);
		assert(neighbor.getIndice(1) == second[neighborhood.position()]);
		i++;
	}

	id = 0;
	for (unsigned int i = 0; i < 3; i++)
		for (unsigned int j = i + 1; j < 4; j++)
			for (unsigned int k = j + 1; k < 5; k++)
				for (unsigned int l = k + 1; l < 6; l++) {
					first[id] = i;
					second[id] = j;
					third[id] = k;
					forth[id] = l;
					id++;
				}

	//test Constructor
	Neighborhood neighborhood1(15, 4);
	//test x-change getter
	assert(neighborhood1.getXChange() == 4);
	//test neighborhood1Size getter
	assert(neighborhood1.getNeighborhoodSize() == 15);
	//test current index getter
	assert(neighborhood1.position() == 0);

	//test neighborhood methods

	//test hasNeighbor
	assert(neighborhood1.hasNeighbor(sol) == true);

	//test init
	neighborhood1.init(sol, neighbor1);
	assert(neighbor1.index() == 0);
	assert(neighbor1.getXChange() == 4);
	assert(neighborhood1.position() == 0);

	//test Mapping
	assert(neighbor1.getIndice(0) == first[neighborhood1.position()]);
	assert(neighbor1.getIndice(1) == second[neighborhood1.position()]);
	assert(neighbor1.getIndice(2) == third[neighborhood1.position()]);
	assert(neighbor1.getIndice(3) == forth[neighborhood1.position()]);

	//test next & cont
	i = 1;
	while (neighborhood.cont(sol)) {
		neighborhood.next(sol, neighbor1);
		assert(neighborhood.position() == i);
		//test Mapping
		assert(neighbor1.getIndice(0) == first[neighborhood1.position()]);
		assert(neighbor1.getIndice(1) == second[neighborhood1.position()]);
		assert(neighbor1.getIndice(2) == third[neighborhood1.position()]);
		assert(neighbor1.getIndice(3) == forth[neighborhood1.position()]);
		i++;
	}

	delete[] (first);
	delete[] (second);
	delete[] (third);
	delete[] (forth);
	std::cout << "[t-moGPUXChange] => OK" << std::endl;

	return EXIT_SUCCESS;
}

