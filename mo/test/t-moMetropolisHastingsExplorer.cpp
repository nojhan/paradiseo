/*
<t-moMetropolisHastingsExplorer.cpp>
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

#include <explorer/moSimpleMetropolisHastingsExplorer.h>
#include "moTestClass.h"

#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moMetropolisHastingsExplorer] => START" << std::endl;

    //Instanciation
    eoBit<eoMinimizingFitness> sol(4, true);
    sol.fitness(4);
    bitNeighborhood nh(4);
    evalOneMax eval(4);
    moNeighborComparator<bitNeighbor> ncomp;
    moSolNeighborComparator<bitNeighbor> sncomp;

    //moSimpleMetropolisHastingsExplorer<bitNeighbor> test(nh, eval, ncomp, sncomp, 3);
    moSimpleMetropolisHastingsExplorer<bitNeighbor> test(nh, eval, 3, sncomp);

    //test de l'acceptation d'un voisin améliorant
    test.initParam(sol);
    test(sol);
    assert(test.accept(sol));
    test.move(sol);
    assert(sol.fitness()==3);
    test.updateParam(sol);
    assert(test.isContinue(sol));

    unsigned int oui=0, non=0;

    //test de l'acceptation d'un voisin non améliorant
    for (unsigned int i=0; i<1000; i++) {
        test(sol);
        if (test.accept(sol))
            oui++;
        else
            non++;
    }
    std::cout << "Attention test en fonction d'une proba \"p\" uniforme dans [0,1] , oui si p < 3/4, non sinon -> resultat sur 1000 essai" << std::endl;
    std::cout << "oui: " << oui << std::endl;
    std::cout << "non: " << non << std::endl;

    assert(oui > 700 && oui < 800); //verification grossiere

    //test du critere d'arret
    test.updateParam(sol);
    assert(test.isContinue(sol));
    test.updateParam(sol);
    assert(!test.isContinue(sol));

    //test de l'acceptation d'un voisin
    sol[0]=false;
    sol[1]=false;
    sol[2]=false;
    sol[3]=false;
    sol.fitness(0);

    test.initParam(sol);
    test(sol);
    assert(!test.accept(sol));

    std::cout << "[t-moMetropolisHastingsExplorer] => OK" << std::endl;

    return EXIT_SUCCESS;
}

