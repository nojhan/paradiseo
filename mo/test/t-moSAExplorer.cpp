/*
<t-moSAExplorer.cpp>
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

#include <iostream>
#include <cstdlib>
#include <cassert>

#include "moTestClass.h"
#include <explorer/moSAExplorer.h>
#include <coolingSchedule/moSimpleCoolingSchedule.h>

int main() {

    std::cout << "[t-moSAExplorer] => START" << std::endl;

    eoBit<eoMinimizingFitness> sol(4, true);
    sol.fitness(4);
    bitNeighborhood nh(4);
    bitNeighborhood emptyNH(0);
    evalOneMax eval(4);
    moSolNeighborComparator<bitNeighbor> sncomp;
    moSimpleCoolingSchedule<bitVector> cool(10,0.1,2,0.1);

    moSAExplorer<bitNeighbor> test1(emptyNH, eval, cool, sncomp);
    moSAExplorer<bitNeighbor> test2(nh, eval, cool, sncomp);

    //test d'un voisinage vide
    test1.initParam(sol);
    test1(sol);
    assert(!test1.accept(sol));
    assert(test1.getTemperature()==10.0);

    //test d'un voisinage "normal"
    test2.initParam(sol);
    test2(sol);
    assert(test2.accept(sol));
    test2.updateParam(sol);
    assert(test2.isContinue(sol));
    test2.move(sol);
    assert(sol.fitness()==3);
    unsigned int ok=0;
    unsigned int ko=0;
    for (unsigned int i=0; i<1000; i++) {
        test2(sol);
        if (test2.isContinue(sol))
            test2.updateParam(sol);
        if (test2.accept(sol))
            ok++;
        else
            ko++;
        test2.move(sol);
    }
    assert((ok>0) && (ko>0));



    std::cout << "[t-moSAExplorer] => OK" << std::endl;

    return EXIT_SUCCESS;
}

