/*
* <t-moeoASEEA.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Lille-Nord Europe, 2006-2008
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
// t-moeoASEEA.cpp
//-----------------------------------------------------------------------------

#include <paradiseo/eo.h>
#include <paradiseo/moeo.h>
#include <cassert>
#include <paradiseo/eo/es/eoRealInitBounded.h>
#include <paradiseo/eo/es/eoRealOp.h>

class ObjectiveVectorTraits : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int i)
    {
        return true;
    }
    static bool maximizing (int i)
    {
        return false;
    }
    static unsigned int nObjectives ()
    {
        return 2;
    }
};

typedef moeoRealObjectiveVector < ObjectiveVectorTraits > ObjectiveVector;

class Solution : public moeoRealVector < ObjectiveVector, double, double >
{
public:
    Solution() : moeoRealVector < ObjectiveVector, double, double > (1) {}
};

class TestEval : public moeoEvalFunc < Solution >
{
public:
    void operator () (Solution & _sol)
    {
        ObjectiveVector objVec;
        objVec[0] = _sol[0];
        objVec[1] = 1 / (_sol[0]+1);
        _sol.objectiveVector(objVec);
    }
};

int main()
{
    std::cout << "[moeoASSEA] => \n";

    TestEval eval;
    eoQuadCloneOp < Solution > xover;
    eoUniformMutation < Solution > mutation(1.0);

    eoRealVectorBounds bounds(1, 1.0, 2.0);
    eoRealInitBounded < Solution > init(bounds);
    eoPop < Solution > pop(20, init);

    for(unsigned int i=0; i<pop.size(); i++)
    	eval(pop[i]);

    eoSGATransform < Solution > transform(xover, 0.1, mutation, 0.1);
    eoGenContinue <Solution > continuator(10);
	moeoUnboundedArchive < Solution > archive;
    eoPopLoopEval <Solution> loopEval(eval);
    eoPopEvalFunc <Solution>& popEval(loopEval);

    // build ASSEA
	moeoASEEA <Solution> algo1(10, eval, xover, 0.1, mutation, 0.5, archive, 10);
	moeoASEEA <Solution> algo2(continuator, popEval, xover, 0.1, mutation, 1.0, archive, 10);
	moeoASEEA <Solution> algo3(continuator, eval, xover, 0.1, mutation, 1.0, archive, 10);
	moeoASEEA <Solution> algo4(continuator, popEval, transform, archive, 10);
	moeoASEEA <Solution> algo5(continuator, eval, transform, archive, 10);

    // run the algo

    algo1(pop);
    algo2(pop);
    algo3(pop);
    algo4(pop);
    algo5(pop);

	std::cout << " OK\n";
    return EXIT_SUCCESS;
}
