/*
* <t-moeoEasyEA.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
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
// t-moeoEasyEA.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include <es/eoRealInitBounded.h>
#include <es/eoRealOp.h>
#include <moeo>

//-----------------------------------------------------------------------------

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
        objVec[1] = _sol[0] * _sol[0];
        _sol.objectiveVector(objVec);
    }
};

//-----------------------------------------------------------------------------

int main()
{
    std::cout << "[moeoEasyEA]" << std::endl;

    TestEval eval;
    eoPopLoopEval <Solution> loopEval(eval);
    eoPopEvalFunc <Solution> & popEval(loopEval);
    eoQuadCloneOp < Solution > xover;
    eoUniformMutation < Solution > mutation(0.05);
    eoSequentialOp < Solution > op;
    op.add(xover, 1.0);
    op.add(mutation, 1.0);
    eoSGATransform < Solution > transform(xover, 1.0, mutation, 1.0);
    eoRealVectorBounds bounds(1, 1.0, 2.0);
    eoRealInitBounded < Solution > init(bounds);
    eoPop < Solution > pop(20, init);
    eoGenContinue < Solution > term(20);
    moeoDominanceDepthFitnessAssignment < Solution > fitnessAssignment;
    moeoFrontByFrontCrowdingDiversityAssignment < Solution > diversityAssignment;
    moeoFitnessThenDiversityComparator < Solution > comparator;
    moeoDetTournamentSelect < Solution > selectOne(comparator, 2);
    moeoElitistReplacement < Solution > replace(fitnessAssignment, diversityAssignment, comparator);
    eoGeneralBreeder < Solution > breed(selectOne, op);
    eoSelectMany < Solution > select(selectOne, 1.0);

    // build EA
    moeoEasyEA < Solution > algo1 (term, eval, breed, replace, fitnessAssignment, diversityAssignment);
    moeoEasyEA < Solution > algo2 (term, popEval, breed, replace, fitnessAssignment, diversityAssignment);
    moeoEasyEA < Solution > algo3 (term, eval, select, transform, replace, fitnessAssignment, diversityAssignment);
    moeoEasyEA < Solution > algo4 (term, popEval, select, transform, replace, fitnessAssignment, diversityAssignment);

    // run the algo
    algo4(pop);

    // final pop
    std::cout << "Final population" << std::endl;
    std::cout << pop << std::endl;

    std::cout << "[moeoEasyEA] OK" << std::endl;
    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
