/*
<t-moCheckpoint.cpp>
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

#include <continuator/moCheckpoint.h>
#include <continuator/moTrueContinuator.h>
#include <continuator/moSolutionStat.h>
#include "moTestClass.h"

#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moCheckpoint] => START" << std::endl;

    unsigned int a=2;
    unsigned int b=15;
    unsigned int c= 10;
    unsigned int d= 47;

    eoBit<eoMinimizingFitness> s(3);
    s[0]=true;
    s[1]=true;
    s[2]=false;

    s.fitness(17);


    //verification que toutes les operateurs sont bien apellés
    moSolutionStat< eoBit< eoMinimizingFitness > > stat;
    updater1 up1(a);
    updater1 up2(b);
    monitor1 mon1(c);
    monitor2 mon2(d);
    moTrueContinuator< bitNeighbor > cont;

    moCheckpoint< bitNeighbor > test1(cont);
    moCheckpoint< bitNeighbor > test2(cont, 3);

    test1.add(up1);
    test1.add(up2);
    test1.add(mon1);
    test1.add(mon2);
    test1.add(stat);

    test2.add(up1);
    test2.add(up2);
    test2.add(mon1);
    test2.add(mon2);
    test2.add(stat);

    test1.init(s);
    test1(s);
    assert(a==3 && b==16 && c==11 && d==48);
    assert(stat.value()[0]);
    assert(stat.value()[1]);
    assert(!stat.value()[2]);
    assert(stat.value().fitness()==17);

    test1(s);
    assert(a==4 && b==17 && c==12 && d==49);
    assert(stat.value()[0]);
    assert(stat.value()[1]);
    assert(!stat.value()[2]);
    assert(stat.value().fitness()==17);

    s.fitness(4);

    test2.init(s);
    test2(s);
    assert(a==5 && b==18 && c==13 && d==50);
    assert(stat.value()[0]);
    assert(stat.value()[1]);
    assert(!stat.value()[2]);
    assert(stat.value().fitness()==4);

    s.fitness(6);
    test2(s);
    assert(stat.value().fitness()==4);
    test2(s);
    assert(stat.value().fitness()==6);
    test2(s);
    assert(stat.value().fitness()==6);

    test1.lastCall(s);
    assert(a==9 && b==22 && c==17 && d==54);
    test2.lastCall(s);
    assert(a==10 && b==23 && c==18 && d==55);

    assert(test1.className()=="moCheckpoint");
    std::cout << "[t-moCheckpoint] => OK" << std::endl;

    return EXIT_SUCCESS;
}

