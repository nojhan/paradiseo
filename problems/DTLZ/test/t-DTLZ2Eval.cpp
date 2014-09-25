/* 
* <t-DTLZ2Eval.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jeremie Humeau
* 
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#include <iostream>
#include <moeo>
#include <DTLZ.h>
#include <assert.h>
#include <DTLZ2Eval.h>

#define M_PI 3.14159265358979323846

int main(int argc, char **argv)
{		    
	std::vector <bool> bObjectives(3);
	for(unsigned int i=0; i<3 ; i++)
		bObjectives[i]=true;
	moeoObjectiveVectorTraits::setup(3,bObjectives);
	
	std::cout << "Run test: t-DTLZ2EVAL\n";
	DTLZ problem;
	
	double tolerance=1e-9;
	
	//test1 :Verify evaluation of objective vectors with all variables are fixed at 1.0
	std::cout << "\t> test1:\n";
	problem.resize(7);
	problem[0]=1;
	problem[1]=1;
	problem[2]=1;
	problem[3]=1;
	problem[4]=1;
	problem[5]=1;
	problem[6]=1;
	
	DTLZ2Eval eval;
	eval(problem);
	
	double res = problem.objectiveVector()[0];
	assert( (res + tolerance > 0) && (res - tolerance < 0));
	std::cout << "\t\t- objectiveVector[0] OK\n";
	res = problem.objectiveVector()[1];
	assert( (res + tolerance > 0) && (res - tolerance < 0));
	std::cout << "\t\t- objectiveVector[1] OK\n";
	res = problem.objectiveVector()[2];
	assert( res == 2.25);
	std::cout << "\t\t- objectiveVector[2] OK\n";
	
	//test2 :Verify evaluation of objective vectors with all variables are fixed at 0.0
	std::cout << "\t> test2:\n";
	problem[0]=0;
	problem[1]=0;
	problem[2]=0;
	problem[3]=0;
	problem[4]=0;
	problem[5]=0;
	problem[6]=0;
	
	problem.invalidate();
	eval(problem);
	
	res = problem.objectiveVector()[0];
	assert( res == 2.25);
	std::cout << "\t\t- objectiveVector[0] OK\n";
	res = problem.objectiveVector()[1];
	assert( (res + tolerance > 0) && (res - tolerance < 0));
	std::cout << "\t\t- objectiveVector[1] OK\n";
	res = problem.objectiveVector()[2];
	assert( (res + tolerance > 0) && (res - tolerance < 0));
	std::cout << "\t\t- objectiveVector[2] OK\n";
	
	//test3 :Verify evaluation of objective vectors with all variables are fixed at 0.5
		std::cout << "\t> test3:\n";
		problem[0]=0.5;
		problem[1]=0.5;
		problem[2]=0.5;
		problem[3]=0.5;
		problem[4]=0.5;
		problem[5]=0.5;
		problem[6]=0.5;
		
		problem.invalidate();
		eval(problem);
		
		res = problem.objectiveVector()[0];
		assert( res == 0.5);
		std::cout << "\t\t- objectiveVector[0] OK\n";
		res = problem.objectiveVector()[1];
		assert( res == 0.5);
		std::cout << "\t\t- objectiveVector[1] OK\n";
		res = problem.objectiveVector()[2];
		assert( (res + tolerance > sin(M_PI/4)) && (res - tolerance < sin(M_PI/4)));
		std::cout << "\t\t- objectiveVector[2] OK\n";
	
    return EXIT_SUCCESS;
}

