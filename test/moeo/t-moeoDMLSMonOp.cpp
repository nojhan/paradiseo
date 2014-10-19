/*
<t-moeoDMLSMonOp.cpp>
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

#include "moeoTestClass.h"
#include <paradiseo/moeo/algo/moeoPLS1.h>
#include <paradiseo/eo/eoTimeContinue.h>
#include <paradiseo/moeo/archive/moeoUnboundedArchive.h>
#include <paradiseo/moeo/hybridization/moeoDMLSMonOp.h>
#include <paradiseo/moeo/selection/moeoExhaustiveUnvisitedSelect.h>
#include <paradiseo/moeo/explorer/moeoExhaustiveNeighborhoodExplorer.h>

int main(){

	std::cout << "[t-moeoDMLSMonOp] => START" << std::endl;

	//init all components
	moeoUnboundedArchive<Solution> arch(false);
	eoTimeContinue<Solution> cont(1);
	fullEvalSolution fullEval(8);
	Solution s;
	evalSolution eval(8);
	ObjectiveVector o;
	SolNeighbor n;
	SolNeighborhood nh(8);
	moeoPLS1<SolNeighbor> pls1(cont, fullEval, arch, nh, eval);
	moeoExhaustiveUnvisitedSelect<Solution> select;
	moeoExhaustiveNeighborhoodExplorer<SolNeighbor> explorer(nh, eval);

	//Create a solution
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);
	s.push_back(true);

	//Set its objective Vector
	o[0]=8;
	o[1]=0;
	s.objectiveVector(o);

	//test constructor 1 with a dmls and its archive
	moeoDMLSMonOp<SolNeighbor> test1(pls1, arch);

	//test constructor 2 with an incremental evaluation function, a selector and an explorer
	moeoDMLSMonOp<SolNeighbor> test2(fullEval, explorer, select, 2, true);

	//test constructor 3 with an incremental evaluation function, a selector and an explorer and the dmls archive
	moeoDMLSMonOp<SolNeighbor> test3(fullEval, arch, explorer, select, 2, true);

	std::cout << "initial solution:"  << std::endl;
	std::cout << s << std::endl;

	test1(s);

	std::cout << "mutate solution:"  << std::endl;
	std::cout << s << std::endl;

	std::cout << "[t-moeoDMLSMonOp] => OK" << std::endl;

	return EXIT_SUCCESS;
}

