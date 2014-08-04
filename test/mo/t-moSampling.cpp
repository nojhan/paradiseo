/*
<t-moSampling.cpp>
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

#include <paradiseo/mo/sampling/moSampling.h>
#include "moTestClass.h"
#include <paradiseo/mo/algo/moFirstImprHC.h>
#include <paradiseo/problems/eval/oneMaxEval.h>
#include <paradiseo/mo/continuator/moSolutionStat.h>
#include <paradiseo/mo/continuator/moCounterStat.h>
#include <paradiseo/mo/continuator/moIterContinuator.h>

int main() {

    std::cout << "[t-moSampling] => START" << std::endl;

    bitNeighborhood nh(4);
    oneMaxEval<bitVector> fullEval;
    evalOneMax eval(4);
    dummyInit2 init(4);
    moIterContinuator<bitNeighbor> cont(3);

    moFirstImprHC<bitNeighbor> hc(nh, fullEval, eval, cont);
    moSolutionStat<bitVector> stat1;
    moCounterStat<bitVector> stat2;
    moSampling<bitNeighbor> test(init, hc, stat1);

    test.add(stat2);

    test();

    std::vector<double> res;
    std::vector<bitVector> res2;
    res = test.getValues(1);
    res2= test.getSolutions(0);

    for (unsigned int i=0; i<res2.size(); i++)
        assert(res2[i].fitness()==4-i);

    for (unsigned int i=0; i<res.size(); i++)
        assert(res[i]==i);

    test.fileExport("outputTestSampling1");
    test.fileExport(1, "outputTestSampling2");

    assert(test.className()=="moSampling");

    std::cout << "[t-moSampling] => OK" << std::endl;

    return EXIT_SUCCESS;
}

