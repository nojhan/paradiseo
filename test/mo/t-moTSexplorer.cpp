/*
<t-moTSexplorer.cpp>
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
#include <paradiseo/mo/memory/moDummyIntensification.h>
#include <paradiseo/mo/memory/moDummyDiversification.h>
#include <paradiseo/mo/memory/moBestImprAspiration.h>
#include <paradiseo/mo/explorer/moTSexplorer.h>
#include "moTestClass.h"

#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moTSexplorer] => START" << std::endl;

    //instansiation
    eoBit<eoMinimizingFitness> sol(4, true);
    sol.fitness(4);
    bitNeighborhood nh(4);
    bitNeighborhood emptyNH(0);
    evalOneMax eval(4);
    moNeighborComparator<bitNeighbor> ncomp;
    moSolNeighborComparator<bitNeighbor> sncomp;
    moDummyIntensification<bitNeighbor> intens;
    moDummyDiversification<bitNeighbor> diver;
    moSolVectorTabuList<bitNeighbor> tabuList(4,0);
    moBestImprAspiration<bitNeighbor> aspir;

    moTSexplorer<bitNeighbor> test(nh, eval, ncomp, sncomp, tabuList, intens, diver, aspir);
    moTSexplorer<bitNeighbor> test2(emptyNH, eval, ncomp, sncomp, tabuList, intens, diver, aspir);

    //test d'un voisinage vide
    test2.initParam(sol);
    test2(sol);
    assert(!test2.accept(sol));

    //test le comportement classique de la taboo
    test.initParam(sol);
    assert(aspir.getBest()==sol);

    test(sol);
    test.updateParam(sol);
    assert(aspir.getBest()==sol);

    //on ameliore et on stock une sol tabou 0111
    test(sol);
    test.move(sol);
    test.moveApplied(true);
    test.updateParam(sol);
    assert(aspir.getBest()==sol);

    //on ameliore et on stock une autre sol tabou 0011
    test(sol);
    test.move(sol);
    test.moveApplied(true);
    test.updateParam(sol);
    assert(aspir.getBest()==sol);

    //pareil on stock 0001 met pdt la recherche on se rend compte que 0111 est tabou
    test(sol);
    test.move(sol);
    test.moveApplied(true);
    test.updateParam(sol);
    assert(aspir.getBest()==sol);

    //on modifie la sol en 1001(fitness 2) pour que la 1er sol exploré(0001) soit tabou
    //De plus on change la solution mais elle est pas meilleure que la best so Far
    sol[0]=true;
    std::cout << sol << std::endl;
    sol.fitness(2);
    test(sol);
    test.move(sol);
    test.moveApplied(true);
    test.updateParam(sol);
    assert(	sol[0] && !sol[1] && !sol[2] && !sol[3]);
    sol[0]=false;
    sol[3]=true;
    assert(aspir.getBest()==sol);

    //test du isContinue
    assert(test.isContinue(sol));

    //test du terminate
    test.initParam(sol);
    sol[0]=true;
    sol[1]=true;
    sol[2]=true;
    sol[3]=true;
    sol.fitness(4);
    test(sol);
    test.move(sol);
    test.moveApplied(true);
    test.updateParam(sol);
    assert(	!sol[0] && sol[1] && sol[2] && sol[3]);
    test.terminate(sol);
    assert(	!sol[0] && !sol[1] && !sol[2] && sol[3]);

    //test pour avoir que des mouvement taboo
    eoBit<eoMinimizingFitness> sol2(2, true);
    sol2.fitness(2);
    bitNeighborhood nh2(2);
    evalOneMax eval2(2);

    moTSexplorer<bitNeighbor> test3(nh2, eval2, ncomp, sncomp, tabuList, intens, diver, aspir);

    test3.initParam(sol2);
    test3(sol2);
    test3.move(sol2);
    test3.moveApplied(true);
    test3.updateParam(sol2);

    test3(sol2);
    test3.move(sol2);
    test3.moveApplied(true);
    test3.updateParam(sol2);

    test3(sol2);
    test3.move(sol2);
    test3.moveApplied(true);
    test3.updateParam(sol2);

    test3(sol2);
    test3.move(sol2);
    test3.moveApplied(true);
    test3.updateParam(sol2);

    //on a rempli la liste tabu pour que tout les voisins soit tabu
    test3(sol2);
    assert(!test3.accept(sol2));


    std::cout << "[t-moTSexplorer] => OK" << std::endl;

    return EXIT_SUCCESS;
}

