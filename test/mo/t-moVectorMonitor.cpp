/*
<t-moVectorMonitor.cpp>
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

#include <paradiseo/mo/continuator/moVectorMonitor.h>
#include "moTestClass.h"

int main() {

    std::cout << "[t-moVectorMonitor] => START" << std::endl;


    eoValueParam<double> doubleParam;
    eoValueParam<unsigned int> intParam;
    eoValueParam<bitVector> eotParam;
    eoValueParam<std::string> stringParam;
    doubleParam.value()= 3.1;
    intParam.value()=6;
    bitVector sol(4,true);
    sol.fitness(0);
    eotParam.value()=sol;

    moVectorMonitor<bitVector> test1(doubleParam);
    moVectorMonitor<bitVector> test2(intParam);
    moVectorMonitor<bitVector> test3(eotParam);
    moVectorMonitor<bitVector> test4(stringParam);

    assert(!test1.solutionType());
    assert(!test2.solutionType());
    assert(test3.solutionType());

    test1();
    test2();
    test3();
    doubleParam.value()= 3.3;
    intParam.value()=7;
    sol.fitness(1);
    eotParam.value()=sol;
    test1();
    test2();
    test3();
    doubleParam.value()= 3.6;
    intParam.value()=8;
    sol.fitness(2);
    eotParam.value()=sol;
    test1();
    test2();
    test3();
    doubleParam.value()= 3.001;
    intParam.value()=9;
    test1();
    test2();

    assert(test1.size()==4);
    assert(test2.size()==4);
    assert(test3.size()==3);
    assert(test1.getValue(1)=="3.3");
    assert(test2.getValue(2)=="8");
    std::cout << test3.getValue(2) << std::endl;
    assert(test1.getValues()[0]==3.1);
    assert(test2.getValues()[2]==8);
    assert(test3.getSolutions()[0].fitness()==0);

    test1.fileExport("outputTestVectorMonitor.txt");

    test1.clear();
    test2.clear();
    test3.clear();
    assert(test1.size()==0);
    assert(test2.size()==0);
    assert(test3.size()==0);



    assert(test1.className()=="moVectorMonitor");



    std::cout << "[t-moVectorMonitor] => OK" << std::endl;

    return EXIT_SUCCESS;
}

