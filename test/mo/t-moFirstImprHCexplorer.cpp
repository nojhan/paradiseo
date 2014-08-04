/*
<t-moFirstImprExplorer.cpp>
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

#include <paradiseo/mo/comparator/moNeighborComparator.h>
#include <paradiseo/mo/comparator/moSolNeighborComparator.h>
#include <paradiseo/mo/explorer/moFirstImprHCexplorer.h>
#include "moTestClass.h"

#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moFirstImprHCexplorer] => START" << std::endl;

    //Instanciation
    eoBit<eoMinimizingFitness> sol(4, true);
    sol.fitness(4);
    bitNeighborhood nh(4);
    evalOneMax eval(4);
    moNeighborComparator<bitNeighbor> ncomp;
    moSolNeighborComparator<bitNeighbor> sncomp;

    moFirstImprHCexplorer<bitNeighbor> test(nh, eval, ncomp, sncomp);

    //on verifie qu'on améliore peut continuer à explorer tant qu'on améliore la solution

    test(sol);
    assert(test.accept(sol));
    test.move(sol);
    assert(sol.fitness()==3);
    assert(!sol[0]);
    assert(test.isContinue(sol));


    test(sol);
    assert(test.accept(sol));
    test.move(sol);
    assert(sol.fitness()==2);
    assert(!sol[1]);
    assert(test.isContinue(sol));

    test(sol);
    assert(test.accept(sol));
    test.move(sol);
    assert(sol.fitness()==1);
    assert(!sol[2]);
    assert(test.isContinue(sol));

    test(sol);
    assert(test.accept(sol));
    test.move(sol);
    assert(sol.fitness()==0);
    assert(!sol[3]);
    assert(test.isContinue(sol));

    test(sol);
    assert(!test.accept(sol));
    assert(sol.fitness()==0);
    assert(!test.isContinue(sol));

    std::cout << "[t-moFirstImprHCexplorer] => OK" << std::endl;

    return EXIT_SUCCESS;
}

