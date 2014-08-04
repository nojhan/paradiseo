/*
<t-moCountMoveMemory.cpp>
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

#include <paradiseo/mo/memory/moCountMoveMemory.h>
#include "moTestClass.h"

int main() {

    std::cout << "[t-moCountMoveMemory] => START" << std::endl;

    bitVector sol;
    bitNeighbor n;


    moCountMoveMemory<bitNeighbor> test;

    test.init(sol);
    assert(test.getNbMove()==0);
    assert(test.getNbNoMove()==0);
    assert(test.getCounter()==0);

    test.add(sol,n);
    assert(test.getNbMove()==1);
    assert(test.getNbNoMove()==0);
    assert(test.getCounter()==0);

    test.add(sol,n);
    assert(test.getNbMove()==2);
    assert(test.getNbNoMove()==0);
    assert(test.getCounter()==0);

    test.update(sol,n);
    assert(test.getNbMove()==2);
    assert(test.getNbNoMove()==1);
    assert(test.getCounter()==1);

    test.add(sol,n);
    assert(test.getNbMove()==3);
    assert(test.getNbNoMove()==1);
    assert(test.getCounter()==0);

    test.update(sol,n);
    assert(test.getNbMove()==3);
    assert(test.getNbNoMove()==2);
    assert(test.getCounter()==1);

    test.update(sol,n);
    assert(test.getNbMove()==3);
    assert(test.getNbNoMove()==3);
    assert(test.getCounter()==2);

    test.clearMemory();
    assert(test.getNbMove()==0);
    assert(test.getNbNoMove()==0);
    assert(test.getCounter()==0);

    std::cout << "[t-moCountMoveMemory] => OK" << std::endl;

    return EXIT_SUCCESS;
}

