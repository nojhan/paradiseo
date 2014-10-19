/*
<t-moRandomNeutralWalkExplorer.cpp>
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

#include <paradiseo/mo/explorer/moRandomNeutralWalkExplorer.h>
#include "moTestClass.h"

#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moRandomNeutralWalkExplorer] => START" << std::endl;

    eoBit<eoMinimizingFitness> sol(4, true);
    sol.fitness(4);
    bitNeighborhood nh(4);
    evalOneMax eval(4);
    dummyEvalOneMax eval2(4);
    moSolNeighborComparator<bitNeighbor> sncomp;

    //test avec la fonction d'eval classique
    //on verifie qu'on ne trouve pas de voisin de mm fitness
    moRandomNeutralWalkExplorer<bitNeighbor> test(nh, eval, sncomp, 3);

    test.initParam(sol);
    test(sol);
    assert(!test.accept(sol));
    assert(!test.isContinue(sol));

    //test avec une fonction d'eval bidon qui renvoie toujours la mm fitness
    //on peut donc verifier qu'on s'arette au bout des 3 itérations.
    moRandomNeutralWalkExplorer<bitNeighbor> test2(nh, eval2, sncomp, 3);

    sol.fitness(2);
    test2.initParam(sol);
    test2(sol);
    assert(test2.accept(sol));
    test2.move(sol);
    assert(sol.fitness()==2);
    test2.updateParam(sol);
    assert(test2.isContinue(sol));

    test2(sol);
    assert(test2.accept(sol));
    test2.move(sol);
    assert(sol.fitness()==2);
    test2.updateParam(sol);
    assert(test2.isContinue(sol));

    test2(sol);
    assert(test2.accept(sol));
    test2.move(sol);
    assert(sol.fitness()==2);
    test2.updateParam(sol);
    assert(!test2.isContinue(sol));

    std::cout << "[t-moRandomNeutralWalkExplorer] => OK" << std::endl;

    return EXIT_SUCCESS;
}

