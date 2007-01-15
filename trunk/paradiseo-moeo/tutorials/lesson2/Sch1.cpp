// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// Sch1.cpp
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

using namespace std;

#include <stdio.h>

/* ParadisEO-EO */
#include <eo>
#include <es.h>

/* ParadisEO-MOEO */
#include <moeoNSGA_II.h>
#include <moeoArchive.h>


// Extend eoParetoFitnessTraits
class SCH1Traits:public eoParetoFitnessTraits
{
  public:static bool maximizing (int i)
  {
    return false;
  }				// is the i-th objective                                                                 
  static unsigned nObjectives ()
  {
    return 2;
  }				// number of objectives
};

// Code decision variables
typedef eoParetoFitness < SCH1Traits > SCH1Fit;

class SCH1EO:public eoReal < SCH1Fit >
{
public:
  SCH1EO ():eoReal < SCH1Fit > (1)
  {
  }
};

// evaluation of the individuals
class SCH1Eval:public eoEvalFunc < SCH1EO >
{
public:
  SCH1Eval ():eoEvalFunc < SCH1EO > ()
  {
  }

  void operator () (SCH1EO & _eo)
  {
    SCH1Fit fitness;
    double x = _eo[0];

    fitness[0] = x * x;
    fitness[1] = (x - 2.0) * (x - 2.0);

    _eo.fitness (fitness);
  }
};

int
main (int argc, char *argv[])
{

  unsigned POP_SIZE = 20;
  unsigned MAX_GEN = 100;
  double M_EPSILON = 0.01;
  double P_CROSS = 0.25;
  double P_MUT = 0.35;

  // The fitness evaluation
  SCH1Eval eval;

  // choose crossover and mutation
  eoQuadCloneOp < SCH1EO > xover;
  eoUniformMutation < SCH1EO > mutation (M_EPSILON);

  // generate initial population
  eoRealVectorBounds bounds (1, 0.0, 2.0);	// [0, 2]
  eoRealInitBounded < SCH1EO > init (bounds);
  eoPop < SCH1EO > pop (POP_SIZE, init);

  // pass parameters to NSGA2
  moeoNSGA_II < SCH1EO > nsga2 (MAX_GEN, eval, xover, P_CROSS, mutation,
				P_MUT);

  // run the algo
  nsga2 (pop);

  // extract first front of the final population (this is the solution of nsga2)
  moeoArchive < SCH1EO > arch;
  arch.update (pop);

  // printing of the final archive
  cout << "Final Archive\n";
  arch.sortedPrintOn (cout);
  cout << endl;

  return EXIT_SUCCESS;
}
