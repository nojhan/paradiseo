/*
<t-moSolVectorTabuList.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#include <memory/moSolVectorTabuList.h>
#include "moTestClass.h"

#include <iostream>
#include <cstdlib>
#include <cassert>

int main(){

	std::cout << "[t-moSolVectorTabuList] => START" << std::endl;

	moSolVectorTabuList<bitNeighbor> test(3);
	bitNeighbor n;

	eoBit<eoMinimizingFitness> sol1(4);
	eoBit<eoMinimizingFitness> sol2(4);
	eoBit<eoMinimizingFitness> sol3(4);
	eoBit<eoMinimizingFitness> sol4(4);

	sol2[0]=true;
	sol3[1]=true;
	sol4[2]=true;

	test.init(sol1);
	test.add(sol1,n);
	test.add(sol2,n);
	test.add(sol3,n);
	assert(test.check(sol1,n));
	assert(test.check(sol2,n));
	assert(test.check(sol3,n));
	test.add(sol4,n);
	assert(!test.check(sol1,n));
	assert(test.check(sol2,n));
	assert(test.check(sol3,n));
	assert(test.check(sol4,n));

	test.init(sol1);
	assert(!test.check(sol1,n));
	assert(!test.check(sol2,n));
	assert(!test.check(sol3,n));
	assert(!test.check(sol4,n));

	test.update(sol1,n);


	std::cout << "[t-moSolVectorTabuList] => OK" << std::endl;

	return EXIT_SUCCESS;
}

