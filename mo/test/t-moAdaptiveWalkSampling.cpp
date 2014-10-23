/*
<t-moAdaptiveWalkSampling.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Boufaras Karima

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

#include <sampling/moAdaptiveWalkSampling.h>
#include "moTestClass.h"
#include <algo/moFirstImprHC.h>
#include <eval/oneMaxEval.h>
#include <continuator/moSolutionStat.h>
#include <continuator/moCounterStat.h>
#include <continuator/moIterContinuator.h>

int main() {

    std::cout << "[t-moAdaptiveWalkSampling] => START" << std::endl;

    bitNeighborhood nh(5);
    oneMaxEval<bitVector> fullEval;
    evalOneMax eval(5);
    dummyInit2 init(5);
    moIterContinuator<bitNeighbor> cont(2);

    moAdaptiveWalkSampling<bitNeighbor> test(init,nh,fullEval,eval,3);
    moCounterStat<bitVector> stat2;
    test.add(stat2);

    test();

    std::vector<double> res;
    std::vector<bitVector> res2;
    res = test.getValues(2);
    res2= test.getSolutions(0);

    for (unsigned int i=0; i<res2.size(); i++)
        assert(res2[i].fitness()==5-i);

    for (unsigned int i=0; i<res.size(); i++)
        assert(res[i]==i);

    test.fileExport("outputTestAdaptativeWalkSampling1");
    test.fileExport(1, "outputTestAdaptativeWalkSampling2");


    std::cout << "[t-moAdaptiveWalkSampling] => OK" << std::endl;

    return EXIT_SUCCESS;
}

