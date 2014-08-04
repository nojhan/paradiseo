/*
<t-moRndWithoutReplNeighborhood.cpp>
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

#include <paradiseo/mo/neighborhood/moRndWithoutReplNeighborhood.h>

#include "moTestClass.h"
#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moRndWithoutReplNeighborhood] => START" << std::endl;

    unsigned int a, b, c;
    eoBit<int> sol;
    moBitNeighbor<int> n;

    //instanciation
    moRndWithoutReplNeighborhood< moBitNeighbor<int> > test(3);
    moRndWithoutReplNeighborhood< moBitNeighbor<int> > test2(0);

    //on verifie que test a bien des voisins et que test2 n'en a pas
    assert(test.hasNeighbor(sol));
    assert(!test2.hasNeighbor(sol));

    //on recupere successivement les index
    test.init(sol, n);
    assert(test.cont(sol));
    a=test.position();
    test.next(sol, n);
    assert(test.cont(sol));
    b=test.position();
    test.next(sol,n);
    assert(!test.cont(sol));
    c=test.position();

    //on s'assure qu'on a bien 0, 1 et 2 (dans un ordre aléatoire)
    assert(a==0 || b==0 || c==0);
    assert(a==1 || b==1 || c==1);
    assert(a==2 || b==2 || c==2);

    assert(test.className()=="moRndWithoutReplNeighborhood");

    std::cout << "[t-moRndWithoutReplNeighborhood] => OK" << std::endl;

    return EXIT_SUCCESS;
}

