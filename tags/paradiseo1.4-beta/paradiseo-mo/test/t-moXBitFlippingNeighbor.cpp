/*
 <t-moXBitFlippingNeighbor.cpp>
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
#include <neighborhood/moXBitFlippingNeighbor.h>
#include <eo>
#include <ga/eoBit.h>

typedef eoBit<eoMaximizingFitness> Solution;
typedef moXBitFlippingNeighbor<Solution> Neighbor;

int main() {

	std::cout << "[t-moXBitFlippingNeighbor] => START" << std::endl;

	Solution sol(5, 0);
	for (int i = 0; i < 5; i++) {
		sol[i] = (i % 2 == 0) ? 0 : 1;
	}

	assert(sol[0] == 0);
	assert(sol[1] == 1);
	assert(sol[2] == 0);
	assert(sol[3] == 1);
	assert(sol[4] == 0);

	//test constructor
	Neighbor neighbor;
	assert(neighbor.index() == 0);
	//test x-change getter
	assert(neighbor.getXChange() == 0);

	//test x-change setter
	neighbor.setXChange(2);
	assert(neighbor.getXChange() == 2);

	//test index setter
	for (unsigned int i = 0; i < neighbor.getXChange(); i++)
		neighbor.setIndice(i, i);

	//test index getter
	for (unsigned int i = 0; i < neighbor.getXChange(); i++)
		assert(neighbor.getIndice(i) == i);

	//test move
	neighbor.move(sol);
	assert(sol[neighbor.getIndice(0)] == 1);
	assert(sol[neighbor.getIndice(1)] == 0);

	//test moveBack
	neighbor.moveBack(sol);
	assert(sol[neighbor.getIndice(0)] == 0);
	assert(sol[neighbor.getIndice(1)] == 1);

	//test set & get indice
	 neighbor.setIndice(0, 1);
	 neighbor.setIndice(1, 2);
	 assert(neighbor.getIndice(0) == 1);
	 assert(neighbor.getIndice(1) == 2);

	 //test move
	 neighbor.move(sol);
	 assert(sol[neighbor.getIndice(0)] == 0);
	 assert(sol[neighbor.getIndice(1)] == 1);

	 //test move back
	 neighbor.moveBack(sol);
	 assert(sol[neighbor.getIndice(0)] == 1);
	 assert(sol[neighbor.getIndice(1)] == 0);

	 Neighbor neighbor1(3);

	 //test setter of one index
	 neighbor1.setIndice(0, 2);
	 neighbor1.setIndice(1, 3);
	 neighbor1.setIndice(2, 4);

	 //test getter of one index
	 assert(neighbor1.getIndice(0) == 2);
	 assert(neighbor1.getIndice(1) == 3);
	 assert(neighbor1.getIndice(2) == 4);

	 //test move
	 neighbor1.move(sol);
	 assert(sol[neighbor1.getIndice(0)] == 1);
	 assert(sol[neighbor1.getIndice(1)] == 0);
	 assert(sol[neighbor1.getIndice(2)] == 1);

	 //test moveBack
	 neighbor1.moveBack(sol);
	 assert(sol[neighbor1.getIndice(0)] == 0);
	 assert(sol[neighbor1.getIndice(1)] == 1);
	 assert(sol[neighbor1.getIndice(2)] == 0);

	std::cout << "[t-moXBitFlippingNeighbor] => OK" << std::endl;

	return EXIT_SUCCESS;
}

