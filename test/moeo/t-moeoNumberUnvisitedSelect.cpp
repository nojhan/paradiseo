/*
<t-moeoNumberUnvisitedSelect.cpp>
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

#include <paradiseo/moeo/selection/moeoNumberUnvisitedSelect.h>
#include "moeoTestClass.h"

int main(){

	std::cout << "[t-moeoNumberUnvisitedSelect] => START" << std::endl;

    // objective vectors
    eoPop < Solution > pop;
    pop.resize(5);
    pop[0].flag(1);
    pop[1].flag(0);
    pop[2].flag(0);
    pop[3].flag(1);
    pop[4].flag(0);

    moeoNumberUnvisitedSelect < Solution > select(2);

    std::vector <unsigned int> res;

    //test general
    res=select(pop);
    assert(res.size()==2);
	assert(res[0]==1 || res[0]==2 || res[0]==4);
	assert(res[1]==1 || res[1]==2 || res[1]==4);
	assert(res[0] != res[1]);

	//test au bornes
	moeoNumberUnvisitedSelect < Solution > select2(6);
	res.resize(0);
	res=select2(pop);
	assert(res.size()==3);
	assert(res[0]==1 || res[0]==2 || res[0]==4);
	assert(res[1]==1 || res[1]==2 || res[1]==4);
	assert(res[2]==1 || res[2]==2 || res[2]==4);
	assert(res[0] != res[1]);
	assert(res[0] != res[2]);
	assert(res[1] != res[2]);

	moeoNumberUnvisitedSelect < Solution > select3(0);
	res.resize(0);
	res=select3(pop);
	assert(res.size()==0);


	std::cout << "[t-moeoNumberUnvisitedSelect] => OK" << std::endl;

	return EXIT_SUCCESS;
}

