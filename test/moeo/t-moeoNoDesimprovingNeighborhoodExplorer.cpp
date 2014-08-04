/*
<t-moeoNoDesimprovingNeighborhoodExplorer.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Arnaud Liefooghe, Jérémie Humeau

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

#include "moeoTestClass.h"
#include <paradiseo/moeo/explorer/moeoNoDesimprovingNeighborhoodExplorer.h>

int main(){

	std::cout << "[t-moeoNoDesimprovingNeighborhoodExplorer] => START" << std::endl;

	//init all components
	Solution s;
	evalSolution eval(8, -1);
	ObjectiveVector o;
	SolNeighbor n;
	SolNeighborhood nh(8);
	moeoNoDesimprovingNeighborhoodExplorer<SolNeighbor> explorer(nh, eval);

	//create source and destination population
	eoPop<Solution> src;
	eoPop<Solution> dest;

	//create a vector for selection
	std::vector<unsigned int> v;
	v.push_back(0);

	//create a solution
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);

	//set its objective vector
	o[0]=8;
	o[1]=0;
	s.objectiveVector(o);

	// aplly a move on th solution
	n.index(3);
	eval(s,n);
	n.move(s);
	s.objectiveVector(n.fitness());

	//print initial sol
	std::cout << "solution:" << std::endl;
	std::cout << s << std::endl;

	//copy the solution in the source population
	src.push_back(s);

	//test explorer, the evaluation function is adapt to test this explorer
	explorer(src, v, dest);

	//verify destination population
	assert(dest.size()==1);
	assert(dest[0].objectiveVector()[0]==10);
	assert(dest[0].objectiveVector()[1]==0);

	std::cout << "destination:" << std::endl;
	for(unsigned int i=0; i<dest.size(); i++)
		std::cout << dest[i] << std::endl;

	std::cout << "[t-moeoNoDesimprovingNeighborhoodExplorer] => OK" << std::endl;

	return EXIT_SUCCESS;
}

