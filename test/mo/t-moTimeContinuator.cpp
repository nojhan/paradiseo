/*
<t-moTimeContinuator.cpp>
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

#include <paradiseo/mo/continuator/moTimeContinuator.h>
#include "moTestClass.h"
#include <time.h>

void wait ( int seconds )
{
    clock_t endwait;
    endwait = clock () + seconds * CLOCKS_PER_SEC ;
    while (clock() < endwait) {}
}


int main() {

    std::cout << "[t-moTimeContinuator] => START" << std::endl;

    moTimeContinuator<moDummyNeighborTest> test(2, false);
    moTimeContinuator<moDummyNeighborTest> test2(3);
    Solution s;

    test.init(s);
    assert(test(s));
    wait(1);
    assert(test(s));
    wait(1);
    assert(!test(s));
    test.init(s);
    assert(test(s));
    wait(2);
    assert(!test(s));

    assert(test.className()=="moTimeContinuator");

    std::cout << "[t-moTimeContinuator] => OK" << std::endl;

    return EXIT_SUCCESS;
}

