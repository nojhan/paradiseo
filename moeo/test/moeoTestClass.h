/*
  <moeoTestClass.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.

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

#ifndef _moeoTestClass_h
#define _moeoTestClass_h

#include <moeo>
#include <EO.h>
#include <eoEvalFunc.h>
#include <neighborhood/moNeighbor.h>
#include <neighborhood/moBackableNeighbor.h>
#include <neighborhood/moNeighborhood.h>
#include <neighborhood/moRndNeighborhood.h>
#include <eval/moEval.h>

#include <ga/eoBit.h>
#include <eoScalarFitness.h>
#include <neighborhood/moOrderNeighborhood.h>
#include <problems/bitString/moBitNeighbor.h>

class ObjectiveVectorTraits : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int /*i*/)
    {
        return true;
    }
    static bool maximizing (int /*i*/)
    {
        return false;
    }
    static unsigned int nObjectives ()
    {
        return 2;
    }
};

typedef moeoRealObjectiveVector < ObjectiveVectorTraits > ObjectiveVector;

typedef moeoBitVector<ObjectiveVector> Solution;

class SolNeighbor : public moIndexNeighbor <Solution, ObjectiveVector>
{
public:
    virtual void move(Solution & _solution) {
        _solution[key] = !_solution[key];
    }

};

typedef moOrderNeighborhood<SolNeighbor> SolNeighborhood;

class evalSolution : public moEval< SolNeighbor >
{
private:
    unsigned size;
    int flag;

public:
    evalSolution(unsigned _size, int _flag=1) : size(_size), flag(_flag) {};

    ~evalSolution(void) {} ;

    void operator() (Solution& _sol, SolNeighbor& _n){
    	ObjectiveVector objVec=_sol.objectiveVector();
    	if(flag>0){
			if (_sol[_n.index()]){
				objVec[0]--;
				objVec[1]++;
			}
			else{
				objVec[0]++;
				objVec[1]--;
			}
			_n.fitness(objVec);
    	}
    	else if(flag==0){
			if (_sol[_n.index()]){
				objVec[0]--;
				objVec[1]--;
			}
			else{
				objVec[0]++;
				objVec[1]--;
			}
			_n.fitness(objVec);
    	}
    	else{
			if (_sol[_n.index()]){
				objVec[0]++;
				objVec[1]++;
			}
			else{
				objVec[0]++;
				objVec[1]--;
			}
			_n.fitness(objVec);
    	}
    }

};

class fullEvalSolution : public eoEvalFunc< Solution >
{
private:
    unsigned size;

public:
    fullEvalSolution(unsigned _size) : size(_size){};

    ~fullEvalSolution(void) {} ;

    void operator() (Solution& _sol){
    	ObjectiveVector o;
    	for(unsigned int i=0; i<size; i++)
    		if(_sol[i])
    			o[0]++;
    		else
    			o[1]++;
    	_sol.objectiveVector(o);
    }

};



#endif
