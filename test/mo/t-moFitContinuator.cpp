/*
<t-moFitContinuator.cpp>
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

#include <paradiseo/mo/continuator/moFitContinuator.h>
#include "moTestClass.h"

int main() {

    std::cout << "[t-moFitContinuator] => START" << std::endl;

    moFitContinuator<bitNeighbor> test1(3);
    moFitContinuator<moDummyNeighborTest> test2(5);

    bitVector sol1;
    Solution sol2;

    sol1.fitness(4);
    assert(test1(sol1));
    sol1.fitness(3);
    assert(!test1(sol1));

    sol2.fitness(3);
    assert(test2(sol2));
    sol2.fitness(5);
    assert(!test2(sol2));

    std::cout << "[t-moFitContinuator] => OK" << std::endl;

    return EXIT_SUCCESS;
}

