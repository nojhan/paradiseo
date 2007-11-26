/*
* <moSA.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)
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

#ifndef __moSA_h
#define __moSA_h

#include <eoOp.h>
#include <eoEvalFunc.h>

#include "moAlgo.h"
#include "moRandMove.h"
#include "moMoveIncrEval.h"
#include "moCoolingSchedule.h"
#include "moSolContinue.h"

#include <math.h>

//! Simulated Annealing (SA)
/*!
  Class that describes a Simulated Annealing algorithm.
*/
template < class M > class moSA:public moAlgo < typename M::EOType >
  {

    //! Alias for the type
    typedef typename M::EOType EOT;

    //! Alias for the fitness
    typedef typename EOT::Fitness Fitness;

  public:

    //! SA constructor
    /*!
       All the boxes used by a SA need to be given.

       \param __move_rand a move generator (generally randomly).
       \param __incr_eval a (generaly) efficient evaluation function 
       \param __cont a stopping criterion.
       \param __init_temp the initial temperature.
       \param __cool_sched a cooling schedule, describes how the temperature is modified.
       \param __full_eval a full evaluation function.
     */
    moSA (moRandMove < M > &__move_rand,
          moMoveIncrEval < M > &__incr_eval,
          moSolContinue < EOT > &__cont,
          double __init_temp,
          moCoolingSchedule & __cool_sched, eoEvalFunc < EOT > &__full_eval):
        move_rand (__move_rand),
        incr_eval (__incr_eval),
        cont (__cont),
        init_temp (__init_temp),
        cool_sched (__cool_sched),
        full_eval (__full_eval)
    {}

    //! function that launches the SA algorithm.
    /*!
       As a moTS or a moHC, the SA can be used for HYBRIDATION in an evolutionary algorithm.

       \param __sol a solution to improve.
       \return TRUE.
     */
    bool operator   ()(EOT & __sol)
    {

      if (__sol.invalid ())
        {
          full_eval (__sol);
        }

      double temp = init_temp;

      M move;

      EOT best_sol = __sol;

      do
        {

          cont.init ();
          do
            {

              move_rand (move);

	      Fitness incremental_fitness = incr_eval (move, __sol);

              Fitness delta_fit = incremental_fitness - __sol.fitness ();

	      if((__sol.fitness() > incremental_fitness ) && (exp (delta_fit / temp) > 1.0))
		{
		  delta_fit = -delta_fit;
		}

              if (incremental_fitness > __sol.fitness() || rng.uniform () < exp (delta_fit / temp))
                {
                  __sol.fitness (incremental_fitness);
                  move (__sol);

                  /* Updating the best solution found
                     until now ? */
                  if (__sol.fitness () > best_sol.fitness ())
		    {
		      best_sol = __sol;
		    }
                }

            }
          while (cont (__sol));

        }
      while (cool_sched (temp));

      __sol = best_sol;

      return true;
    }

  private:

    //! A move generator (generally randomly)
    moRandMove < M > &move_rand;

    //! A (generally) efficient evaluation function.
    moMoveIncrEval < M > &incr_eval;

    //! Stopping criterion before temperature update
    moSolContinue < EOT > &cont;

    //! Initial temperature
    double  init_temp;

    //! The cooling schedule
    moCoolingSchedule & cool_sched;

    //! A full evaluation function.
    eoEvalFunc < EOT > &full_eval;	// Full evaluator.
  };

#endif
