/*
<t-moDistanceStat.cpp>
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

#include <paradiseo/mo/problems/bitString/moBitNeighbor.h>
#include <paradiseo/mo/continuator/moDistanceStat.h>
#include <paradiseo/eo/utils/eoDistance.h>

#include <iostream>
#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moDistanceStat] => START" << std::endl;

    eoBit<int> sol1;
    eoBit<int> sol2;
    eoBit<int> sol3;
    sol1.push_back(true);
    sol1.push_back(false);
    sol1.push_back(true);

    sol2.push_back(true);
    sol2.push_back(true);
    sol2.push_back(false);

    sol3.push_back(true);
    sol3.push_back(true);
    sol3.push_back(true);

    //verification de la stat avec une distance de Hamming

    eoHammingDistance< eoBit<int> > dist;

    moDistanceStat< eoBit<int> > test(dist, sol1);

    test(sol2);
    assert(test.value()==2);
    test(sol3);
    assert(test.value()==1);

    assert(test.className()=="moDistanceStat");
    std::cout << "[t-moDistanceStat] => OK" << std::endl;

    return EXIT_SUCCESS;
}

