/*
<t-moeoPLS2.cpp>
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
#include <paradiseo/moeo/algo/moeoPLS2.h>
#include <paradiseo/eo/eoTimeContinue.h>
#include <paradiseo/moeo/archive/moeoUnboundedArchive.h>

int main(){

	std::cout << "[t-moeoPLS2] => START" << std::endl;

	//init all components
	moeoUnboundedArchive<Solution> arch(false);
	eoTimeContinue<Solution> cont(1);
	fullEvalSolution fullEval(8);
	Solution s;
	evalSolution eval(8);
	ObjectiveVector o;
	SolNeighbor n;
	SolNeighborhood nh(8);
	moeoPLS2<SolNeighbor> test(cont, fullEval, arch, nh, eval);

	//create source population
	eoPop<Solution> src;

	//Create a solution
	s.resize(8);
	s[0]=true;
	s[1]=true;
	s[2]=true;
	s[3]=true;
	s[4]=true;
	s[5]=true;
	s[6]=true;
	s[7]=true;

	//Set its objective Vector
	o[0]=8;
	o[1]=0;
	s.objectiveVector(o);

	//apply a move on the solution and compute new objective vector
	n.index(3);
	eval(s,n);
	n.move(s);
	s.objectiveVector(n.fitness());

	//copy the solution in the source population
	src.push_back(s);

	//test PLS2
	test(src);

	//verify all objective vector was found.
	assert(arch.size()==9);
	assert(arch[0].objectiveVector()[0]==7);
	assert(arch[1].objectiveVector()[0]==6);
	assert(arch[2].objectiveVector()[0]==8);
	assert(arch[3].objectiveVector()[0]==5);
	assert(arch[4].objectiveVector()[0]==4);
	assert(arch[5].objectiveVector()[0]==3);
	assert(arch[6].objectiveVector()[0]==2);
	assert(arch[7].objectiveVector()[0]==1);
	assert(arch[8].objectiveVector()[0]==0);

	assert(arch[0].objectiveVector()[1]==1);
	assert(arch[1].objectiveVector()[1]==2);
	assert(arch[2].objectiveVector()[1]==0);
	assert(arch[3].objectiveVector()[1]==3);
	assert(arch[4].objectiveVector()[1]==4);
	assert(arch[5].objectiveVector()[1]==5);
	assert(arch[6].objectiveVector()[1]==6);
	assert(arch[7].objectiveVector()[1]==7);
	assert(arch[8].objectiveVector()[1]==8);

	//Print
	std::cout << "source:" << std::endl;
	std::cout << src << std::endl;

	std::cout << "archive:" << std::endl;
	for(unsigned int i=0; i<arch.size(); i++)
		std::cout << arch[i] << std::endl;


	std::cout << "[t-moeoPLS2] => OK" << std::endl;

	return EXIT_SUCCESS;
}

