/*
<t-moTS.cpp>
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

#include <paradiseo/mo/algo/moTS.h>
#include "moTestClass.h"
#include <paradiseo/problems/eval/oneMaxEval.h>
#include <paradiseo/mo/continuator/moTrueContinuator.h>
#include <paradiseo/mo/comparator/moSolNeighborComparator.h>
#include <paradiseo/mo/comparator/moNeighborComparator.h>
#include <paradiseo/mo/memory/moSolVectorTabuList.h>

int main() {

    std::cout << "[t-moTS] => START" << std::endl;

    bitNeighborhood nh(4);
    oneMaxEval<bitVector> fullEval;
    evalOneMax eval(4);

    //test first constructor
    moTS<bitNeighbor> test1(nh, fullEval, eval, 1, 7);

    //test second constructor
    moSolVectorTabuList<bitNeighbor> tl(7,0);
    moTS<bitNeighbor> test2(nh, fullEval, eval, 1, tl);

    //test third constructor
    moTrueContinuator<bitNeighbor> cont;
    moSolNeighborComparator<bitNeighbor> sncomp;
    moNeighborComparator<bitNeighbor> ncomp;
    moDummyIntensification<bitNeighbor> intens;
    moDummyDiversification<bitNeighbor> div;
    moBestImprAspiration<bitNeighbor> aspir;

    moTS<bitNeighbor> test3(nh, fullEval, eval, ncomp, sncomp, cont, tl, intens, div, aspir);


    std::cout << "[t-moTS] => OK" << std::endl;

    return EXIT_SUCCESS;
}

