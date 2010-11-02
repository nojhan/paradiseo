/*
 <t-moKswapNeighborhood.cu>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

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
#include <neighborhood/moKswapNeighbor.h>
#include <neighborhood/moKswapNeighborhood.h>
#include <eo>

typedef eoInt<eoMaximizingFitness> Solution;
typedef moKswapNeighbor<Solution> Neighbor;
typedef moKswapNeighborhood<Neighbor> Neighborhood;

int main() {

	std::cout << "[t-moKswapNeighborhood] => START" << std::endl;

	//test factorial
	assert(factorial(10) == 3628800);

	//test sizeMapping
	assert(sizeMapping(6, 1) == 15);
	assert(sizeMapping(6, 2) == 20);
	assert(sizeMapping(6, 3) == 15);
	assert(sizeMapping(6, 4) == 6);

	Solution sol(6);
	Neighbor neighbor(1);

	unsigned int * first;
	unsigned int * second;
	unsigned int * indices;

	indices = new unsigned int[2];
	first = new unsigned int[15];
	second = new unsigned int[15];
	unsigned int id = 0;
	for (unsigned int i = 0; i < 5; i++)
		for (unsigned int j = i + 1; j < 6; j++) {
			first[id] = i;
			second[id] = j;
			id++;
		}

	//test Constructor
	Neighborhood neighborhood(6, 1);
	assert(neighborhood.getSize() == 6);
	assert(neighborhood.getKswap() == 1);
	assert(neighborhood.position() == 0);
	assert(neighborhood.getNeighborhoodSize() == 15);

	//test neighborhood methods

	//test hasNeighbor
	assert(neighborhood.hasNeighbor(sol) == true);

	//test init
	neighborhood.init(sol, neighbor);
	assert(neighbor.index() == 0);
	assert(neighbor.getSize() == 6);
	assert(neighbor.getKswap() == 1);
	assert(neighborhood.position() == 0);

	//test Mapping
	indices[0] = first[neighborhood.position()];
	indices[1] = second[neighborhood.position()];
	assert(neighbor.getIndice(0) == indices[0]);
	assert(neighbor.getIndice(1) == indices[1]);

	//test next & cont
	for (int i = 1; i < 15; i++)
		if (neighborhood.cont(sol)) {
			neighborhood.next(sol, neighbor);
			assert(neighborhood.position() == i);
			//test Mapping
			indices[0] = first[neighborhood.position()];
			indices[1] = second[neighborhood.position()];
			assert(neighbor.getIndice(0) == indices[0]);
			assert(neighbor.getIndice(1) == indices[1]);
		}

	delete[] (indices);
	delete[] (first);
	delete[] (second);
	std::cout << "[t-moKswapNeighborhood] => OK" << std::endl;

	return EXIT_SUCCESS;
}

