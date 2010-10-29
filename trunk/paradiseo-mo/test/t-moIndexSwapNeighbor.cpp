/*
 <t-moIndexSwapNeighbor.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Boufaras Karima, Thé Van Luong

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  ue,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

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
#include <neighborhood/moIndexSwapNeighbor.h>
#include <eo>
#include <eoInt.h>

typedef eoInt<eoMaximizingFitness> Solution;
typedef moIndexSwapNeighbor<Solution> Neighbor;

int main() {

	std::cout << "[t-moIndexswapNeighbor] => START" << std::endl;

	//test default constructor
	Neighbor neighbor0;
	assert(neighbor0.index() == 0);
	assert(neighbor0.getKswap() == 0);

	//test constructor
	Neighbor neighbor(1);
	assert(neighbor.index() == 0);
	assert(neighbor.getKswap() == 1);

	//test setter & getter of one index
	for (unsigned int i = 0; i <= neighbor.getKswap(); i++)
		neighbor.setIndice(i, i);

	for (unsigned int i = 0; i <= neighbor.getKswap(); i++)
		assert(neighbor.getIndice(i) == i);

	//test copy Constructor
	Neighbor neighbor1(neighbor);
	assert(neighbor1.index() == 0);
	assert(neighbor1.getKswap() == 1);
	for (unsigned int i = 0; i <= neighbor1.getKswap(); i++)
		assert(neighbor1.getIndice(i) ==i);

	//test assignement operator

	for (unsigned int i = 0; i <= neighbor1.getKswap(); i++)
			neighbor1.setIndice(i, 0);

	//test setter and getter of set of indexes
	neighbor1.setIndices(neighbor.getIndices());
	for (unsigned int i = 0; i <= neighbor1.getKswap(); i++)
			assert(neighbor1.getIndice(i) ==i);

	Neighbor neighbor2(2);

	//test setter of one index
	neighbor2.setIndice(0, 0);
	neighbor2.setIndice(1, 1);
	neighbor2.setIndice(2, 2);

	//test getter of one index
	assert(neighbor2.getIndice(0) == 0);
	assert(neighbor2.getIndice(1) == 1);
	assert(neighbor2.getIndice(2) == 2);

	std::cout << "[t-moKswapNeighbor] => OK" << std::endl;

	return EXIT_SUCCESS;
}
