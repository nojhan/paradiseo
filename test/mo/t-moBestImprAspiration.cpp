/*
<t-moBestImprAspiration.cpp>
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
#include <paradiseo/mo/memory/moBestImprAspiration.h>
#include "moTestClass.h"

#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moBestImprAspiration] => START" << std::endl;

    moBestImprAspiration<bitNeighbor> test;
    eoBit<eoMinimizingFitness> sol1(4);
    eoBit<eoMinimizingFitness> sol2(4);
    eoBit<eoMinimizingFitness> sol3(4);
    eoBit<eoMinimizingFitness> sol4(4);
    bitNeighbor n1;
    bitNeighbor n2;
    bitNeighbor n3;
    bitNeighbor n4;

    sol3[0]=true;
    sol4[3]=true;

    sol1.fitness(4);
    sol2.fitness(5);
    sol3.fitness(3);
    sol4.fitness(3);
    n1.fitness(4);
    n2.fitness(5);
    n3.fitness(3);
    n4.fitness(3);


    //verification qu'on update bien le best so far quand il faut
    test.init(sol1);
    assert(test.getBest()==sol1);
    assert(!test(sol2,n2));
    assert(test(sol3,n3));
    test.update(sol3,n3);
    assert(test.getBest()==sol3);
    assert(!test(sol4,n4));
    test.update(sol4,n4);
    assert(test.getBest()==sol3);

    std::cout << "[t-moBestImprAspiration] => OK" << std::endl;

    return EXIT_SUCCESS;
}

