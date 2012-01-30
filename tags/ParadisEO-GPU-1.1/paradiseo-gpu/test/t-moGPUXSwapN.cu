/*
 <t-moGPUXSwapN.cu>
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
#include <eo>

typedef eoInt<eoMaximizingFitness> Solution;
typedef moGPUXSwapN<Solution> Neighbor;

int main() {

	std::cout << "[t-moGPUXSwapN] => START" << std::endl;

	Solution sol1(5);
	for (int i = 4; i >=0; i--)
		sol1[4-i] = i;

	//test constructor
	Neighbor neighbor(2);
	assert(neighbor.index() == 0);
	assert(neighbor.getXChange() == 2);

	//test setter of one index
	for (unsigned int i = 0; i < neighbor.getXChange(); i++)
		neighbor.setIndice(i, i);

	//test getter of one index
	for (unsigned int i = 0; i < neighbor.getXChange(); i++)
		assert(neighbor.getIndice(i) == i);

	//test move
	neighbor.move(sol1);
	assert(sol1[neighbor.getIndice(0)] == 3);
	assert(sol1[neighbor.getIndice(1)] == 4);

	//test moveBack
	neighbor.moveBack(sol1);
	assert(sol1[neighbor.getIndice(0)] == 4);
	assert(sol1[neighbor.getIndice(1)] == 3);

	//test set & get indice
	neighbor.setIndice(0, 1);
	neighbor.setIndice(1, 2);
	assert(neighbor.getIndice(0) == 1);
	assert(neighbor.getIndice(1) == 2);

	//test move
	neighbor.move(sol1);
	assert(sol1[neighbor.getIndice(0)] == 2);
	assert(sol1[neighbor.getIndice(1)] == 3);

	//test move back
	neighbor.moveBack(sol1);
	assert(sol1[neighbor.getIndice(0)] == 3);
	assert(sol1[neighbor.getIndice(1)] == 2);

	Neighbor neighbor2(3);

	//test setter of one index
	neighbor2.setIndice(0, 0);
	neighbor2.setIndice(1, 1);
	neighbor2.setIndice(2, 2);

	//test getter of one index
	assert(neighbor2.getIndice(0) == 0);
	assert(neighbor2.getIndice(1) == 1);
	assert(neighbor2.getIndice(2) == 2);

	//test move
	neighbor2.move(sol1);
	assert(sol1[neighbor2.getIndice(0)] == 3);
	assert(sol1[neighbor2.getIndice(1)] == 2);
	assert(sol1[neighbor2.getIndice(2)] == 4);

	//test moveBack
	neighbor2.moveBack(sol1);
	assert(sol1[neighbor2.getIndice(0)] == 4);
	assert(sol1[neighbor2.getIndice(1)] == 3);
	assert(sol1[neighbor2.getIndice(2)] == 2);

	std::cout << "[t-moGPUXSwapN] => OK" << std::endl;

	return EXIT_SUCCESS;
}

