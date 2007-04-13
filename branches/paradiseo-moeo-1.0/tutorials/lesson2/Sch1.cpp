// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// Sch1.cpp
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

using namespace std;

#include <stdio.h>
#include <moeo>
#include <es/eoRealInitBounded.h>
#include <es/eoRealOp.h>
#include <moeoObjectiveVectorTraits.h>

// the moeoObjectiveVectorTraits : minimizing 2 objectives
class Sch1ObjectiveVectorTraits : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int i)
    {
        return true;
    }
    static unsigned nObjectives ()
    {
        return 2;
    }
};


// objective vector of doubles
typedef moeoObjectiveVectorDouble < Sch1ObjectiveVectorTraits > Sch1ObjectiveVector;


// multi-objective evolving object for the Sch1 problem
class Sch1 : public moeoRealVector < Sch1ObjectiveVector, double, double >
{
public:
    Sch1() : moeoRealVector < Sch1ObjectiveVector, double, double > (1) {}
};


// evaluation of objective functions
class Sch1Eval : public moeoEvalFunc < Sch1 >
{
public:
    void operator () (Sch1 & _sch1)
    {
        if (_sch1.invalidObjectiveVector())
        {
            Sch1ObjectiveVector objVec;
            double x = _sch1[0];
            objVec[0] = x * x;
            objVec[1] = (x - 2.0) * (x - 2.0);
            _sch1.objectiveVector(objVec);
        }
    }
};


// main
int main (int argc, char *argv[])
{
    // parameters
    unsigned POP_SIZE = 20;
    unsigned MAX_GEN = 100;
    double M_EPSILON = 0.01;
    double P_CROSS = 0.25;
    double P_MUT = 0.35;

    // objective functions evaluation
    Sch1Eval eval;

    // crossover and mutation
    eoQuadCloneOp < Sch1 > xover;
    eoUniformMutation < Sch1 > mutation (M_EPSILON);

    // generate initial population
    eoRealVectorBounds bounds (1, 0.0, 2.0);	// [0, 2]
    eoRealInitBounded < Sch1 > init (bounds);
    eoPop < Sch1 > pop (POP_SIZE, init);

    // build NSGA-II
    moeoNSGAII < Sch1 > nsgaII (MAX_GEN, eval, xover, P_CROSS, mutation, P_MUT);

    // run the algo
    nsgaII (pop);

    // extract first front of the final population using an moeoArchive (this is the output of nsgaII)
    moeoArchive < Sch1 > arch;
    arch.update (pop);

    // printing of the final archive
    cout << "Final Archive" << endl;
    arch.sortedPrintOn (cout);
    cout << endl;

    return EXIT_SUCCESS;
}
