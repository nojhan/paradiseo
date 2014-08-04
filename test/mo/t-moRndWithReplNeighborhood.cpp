/*
<t-moRndWithReplNeighborhood.cpp>
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

#include <paradiseo/mo/neighborhood/moRndWithReplNeighborhood.h>

#include "moTestClass.h"
#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moRndWithReplNeighborhood] => START" << std::endl;

    unsigned int a, b;
    eoBit<int> sol;
    moBitNeighbor<int> n;

    moRndWithReplNeighborhood< moBitNeighbor<int> > test(3);
    moRndWithReplNeighborhood< moBitNeighbor<int> > test2(0);

    assert(test.hasNeighbor(sol));
    assert(!test2.hasNeighbor(sol));

    test.init(sol,n);

    //on s'assure qu'on a bien toujours bien l'index 0, 1 ou 2 qui est renvoyé
    for (unsigned int i=0; i<100; i++) {

        a=n.index();
        test.next(sol,n);
        b=n.index();

        assert(a==0 || a==1 || a==2);
        assert(b==0 || b==1 || b==2);
        assert(test.cont(sol));
        assert(!test2.cont(sol));
        assert(test.cont(sol));

    }

    assert(test.className()=="moRndWithReplNeighborhood");

    std::cout << "[t-moRndWithReplNeighborhood] => OK" << std::endl;

    return EXIT_SUCCESS;
}

