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

#include <paradiseo/mo/memory/moSolVectorTabuList.h>
#include "moTestClass.h"

#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moSolVectorTabuList] => START" << std::endl;

    //test without countdown
    moSolVectorTabuList<bitNeighbor> test(2,0);
    bitNeighbor n1;
    bitNeighbor n2;
    bitNeighbor n3;
    bitNeighbor n4;
    n1.index(0);
    n2.index(1);
    n3.index(2);
    n4.index(3);

    eoBit<eoMinimizingFitness> sol1(4, true);
    eoBit<eoMinimizingFitness> sol2(4, true);
    eoBit<eoMinimizingFitness> sol3(4, true);
    eoBit<eoMinimizingFitness> sol4(4, true);

    sol2[0]=false;
    sol3[1]=false;
    sol4[0]=false;
    sol4[1]=false;

    //init
    test.init(sol1);

    //ajout d'une sol tabu
    test.add(sol1,n1);

    //verification des voisins de chaques sol
    assert(test.check(sol2,n1));
    assert(!test.check(sol2,n2));
    assert(!test.check(sol2,n3));
    assert(!test.check(sol2,n4));

    assert(!test.check(sol3,n1));
    assert(test.check(sol3,n2));
    assert(!test.check(sol3,n3));
    assert(!test.check(sol3,n4));

    assert(!test.check(sol4,n1));
    assert(!test.check(sol4,n2));
    assert(!test.check(sol4,n3));
    assert(!test.check(sol4,n4));

    test.init(sol1);
    assert(!test.check(sol2,n1));
    assert(!test.check(sol3,n2));

    test.update(sol1,n1);

    test.add(sol1,n1);
    test.add(sol2,n1);
    assert(test.check(sol2,n1));
    test.add(sol4,n1);
    assert(!test.check(sol2,n1));
    assert(test.check(sol2,n2));

    //test with a countdown at 3
    moSolVectorTabuList<bitNeighbor> test2(2,2);
    test2.init(sol1);
    test2.add(sol1,n1);
    assert(test2.check(sol2,n1));
    assert(!test2.check(sol2,n2));
    assert(!test2.check(sol2,n3));
    assert(!test2.check(sol2,n4));

    assert(!test2.check(sol3,n1));
    assert(test2.check(sol3,n2));
    assert(!test2.check(sol3,n3));
    assert(!test2.check(sol3,n4));

    assert(!test2.check(sol4,n1));
    assert(!test2.check(sol4,n2));
    assert(!test2.check(sol4,n3));
    assert(!test2.check(sol4,n4));

    //coutdown sol1 -> 1
    test2.update(sol1,n1);
    assert(test2.check(sol2,n1));
    assert(!test2.check(sol2,n2));
    assert(!test2.check(sol2,n3));
    assert(!test2.check(sol2,n4));

    assert(!test2.check(sol3,n1));
    assert(test2.check(sol3,n2));
    assert(!test2.check(sol3,n3));
    assert(!test2.check(sol3,n4));

    assert(!test2.check(sol4,n1));
    assert(!test2.check(sol4,n2));
    assert(!test2.check(sol4,n3));
    assert(!test2.check(sol4,n4));

    //coutdown sol1 -> 0 : sol1 is no longer tabu
    test2.update(sol1,n1);
    assert(!test2.check(sol2,n1));
    assert(!test2.check(sol2,n2));
    assert(!test2.check(sol2,n3));
    assert(!test2.check(sol2,n4));

    assert(!test2.check(sol3,n1));
    assert(!test2.check(sol3,n2));
    assert(!test2.check(sol3,n3));
    assert(!test2.check(sol3,n4));

    assert(!test2.check(sol4,n1));
    assert(!test2.check(sol4,n2));
    assert(!test2.check(sol4,n3));
    assert(!test2.check(sol4,n4));

    std::cout << "[t-moSolVectorTabuList] => OK" << std::endl;

    return EXIT_SUCCESS;
}

