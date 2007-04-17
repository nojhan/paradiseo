// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopEval.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef _FlowShopEval_h
#define _FlowShopEval_h

// Flow-shop fitness
#include "FlowShopFitness.h"
// include the base definition of eoEvalFunc
#include <eoEvalFunc.h>



/**
 * Functor
 * Computation of the multi-objective evaluation of a FlowShop object
 */
class FlowShopEval:public eoEvalFunc < FlowShop >
{

public:

  /**
   * constructor
   * @param _M the number of machines 
   * @param _N the number of jobs to schedule
   * @param _p the processing times
   * @param _d the due dates
   */
  FlowShopEval (const unsigned _M, const unsigned _N,
		const vector < vector < unsigned > >&_p,
		const vector < unsigned >&_d):M (_M), N (_N), p (_p), d (_d)
  {

    unsigned nObjs = 2;
      std::vector < bool > bObjs (nObjs, false);
      eoVariableParetoTraits::setUp (nObjs, bObjs);
  }



  /**
   * computation of the multi-objective evaluation of an eoFlowShop object
   * @param FlowShop & _eo  the FlowShop object to evaluate
   */
  void operator   () (FlowShop & _eo)
  {
    FlowShopFitness fitness;
    fitness[0] = tardiness (_eo);
    fitness[1] = makespan (_eo);
    _eo.fitness (fitness);
  }





private:

  /** number of machines */
  unsigned M;
  /** number of jobs */
  unsigned N;
  /** p[i][j] = processing time of job j on machine i */
  std::vector < std::vector < unsigned > >p;
  /** d[j] = due-date of the job j */
  std::vector < unsigned >d;



  /**
   * computation of the makespan
   * @param FlowShop _eo  the FlowShop object to evaluate
   */
  double makespan (FlowShop _eo)
  {
    // the scheduling to evaluate
    vector < unsigned >scheduling = _eo.getScheduling ();
    // completion times computation for each job on each machine
    // C[i][j] = completion of the jth job of the scheduling on the ith machine
    std::vector < std::vector < unsigned > >C = completionTime (_eo);
    // fitness == C[M-1][scheduling[N-1]];     
    return C[M - 1][scheduling[N - 1]];
  }



  /**
   * computation of the tardiness
   * @param _eo  the FlowShop object to evaluate
   */
  double tardiness (FlowShop _eo)
  {
    // the scheduling to evaluate
    vector < unsigned >scheduling = _eo.getScheduling ();
    // completion times computation for each job on each machine
    // C[i][j] = completion of the jth job of the scheduling on the ith machine
    std::vector < std::vector < unsigned > >C = completionTime (_eo);
    // tardiness computation
    unsigned long sum = 0;
    for (unsigned j = 0; j < N; j++)
      sum +=
	(unsigned) std::max (0,
			     (int) (C[M - 1][scheduling[j]] -
				    d[scheduling[j]]));
    // fitness == sum
    return sum;
  }



  /**
   * computation of the completion times of a scheduling (for each job on each machine)
   * C[i][j] = completion of the jth job of the scheduling on the ith machine
   * @param const FlowShop _eo  the genotype to evaluate
   */
  std::vector < std::vector < unsigned > >completionTime (FlowShop _eo)
  {
    vector < unsigned >scheduling = _eo.getScheduling ();
    std::vector < std::vector < unsigned > >C (M, N);
    C[0][scheduling[0]] = p[0][scheduling[0]];
    for (unsigned j = 1; j < N; j++)
      C[0][scheduling[j]] = C[0][scheduling[j - 1]] + p[0][scheduling[j]];
    for (unsigned i = 1; i < M; i++)
      C[i][scheduling[0]] = C[i - 1][scheduling[0]] + p[i][scheduling[0]];
    for (unsigned i = 1; i < M; i++)
      for (unsigned j = 1; j < N; j++)
	C[i][scheduling[j]] =
	  std::max (C[i][scheduling[j - 1]],
		    C[i - 1][scheduling[j]]) + p[i][scheduling[j]];
    return C;
  }


};

#endif
