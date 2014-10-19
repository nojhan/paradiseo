/*
<t-moNeighborVectorTabuList.cpp>
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

#include <paradiseo/mo/memory/moNeighborVectorTabuList.h>
#include "moTestClass.h"

#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moNeighborVectorTabuList] => START" << std::endl;

    //tabu list of size 2 (neighbor are always tabu)
    moNeighborVectorTabuList<bitNeighbor> test(2,0);


    bitVector sol(4,true);
    sol.fitness(0);
    bitNeighbor n1;
    bitNeighbor n2;
    bitNeighbor n3;
    bitNeighbor n4;
    n1.index(0);
    n2.index(1);
    n3.index(2);
    n4.index(3);

    //n1 must be tabu
    test.add(sol, n1);
    assert(test.check(sol, n1));
    assert(!test.check(sol, n2));
    assert(!test.check(sol, n3));
    assert(!test.check(sol, n4));

    //n1 and n2 must be tabu
    test.add(sol, n2);
    assert(test.check(sol, n1));
    assert(test.check(sol, n2));
    assert(!test.check(sol, n3));
    assert(!test.check(sol, n4));

    //n3 is added, so it replace n1 -> n2 and n3 must be tabu
    test.add(sol, n3);
    assert(!test.check(sol, n1));
    assert(test.check(sol, n2));
    assert(test.check(sol, n3));
    assert(!test.check(sol, n4));

    //clear tabu list all neighbor must not be tabu
    test.init(sol);
    assert(!test.check(sol, n1));
    assert(!test.check(sol, n2));
    assert(!test.check(sol, n3));
    assert(!test.check(sol, n4));

    //tabu list of size 2 (neighbor are tabu during 2 iterations)
    moNeighborVectorTabuList<bitNeighbor> test2(2,2);

    test2.add(sol, n1);
    assert(test2.check(sol, n1));
    test2.update(sol, n3);
    test2.add(sol,n2);
    assert(test2.check(sol, n1));
    assert(test2.check(sol, n2));
    test2.update(sol, n3);
    assert(!test2.check(sol, n1));
    assert(test2.check(sol, n2));
    test2.update(sol, n4);
    assert(!test2.check(sol, n1));
    assert(!test2.check(sol, n2));

    std::cout << "[t-moNeighborVectorTabuList] => OK" << std::endl;

    return EXIT_SUCCESS;
}

