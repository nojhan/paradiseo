/*
* <moILS.h>
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

#ifndef __moILS_h
#define __moILS_h

#include <eoEvalFunc.h>

#include "moHC.h"
#include "moTS.h"
#include "moSA.h"

//! Iterated Local Search (ILS)
/*!
  Class which describes the algorithm for a iterated local search.
 */
template < class M > class moILS:public moAlgo < typename M::EOType >
  {

    //! Alias for the type.
    typedef typename M::EOType EOT;

    //! Alias for the fitness.
    typedef typename EOT::Fitness Fitness;

  public:

    //! Generic constructor
    /*!
       Generic constructor using a moAlgo

       \param __algo The solution based heuristic to use.
       \param __continue The stopping criterion.
       \param __acceptance_criterion The acceptance criterion.
       \param __perturbation The pertubation generator.
       \param __full_eval The evaluation function.
    */
    moILS (moAlgo<EOT> &__algo, moSolContinue <EOT> &__continue, moComparator<EOT> &__acceptance_criterion, eoMonOp<EOT> &__perturbation,
           eoEvalFunc<EOT> &__full_eval):
        algo(__algo), cont(__continue), acceptance_criterion(__acceptance_criterion), perturbation(__perturbation), full_eval(__full_eval)
    {}

    //! Constructor for using a moHC for the moAlgo
    /*!
      \param __move_init The move initialisation (for the moHC).
      \param __next_move The move generator (for the moHC).
      \param __incr_eval The partial evaluation function (for the moHC).
      \param __move_select The move selection strategy (for the moHC).
      \param __continue The stopping criterion.
      \param __acceptance_criterion The acceptance criterion.
      \param __perturbation The pertubation generator.
      \param __full_eval The evaluation function.
    */
    moILS (moMoveInit < M > &__move_init, moNextMove < M > &__next_move, moMoveIncrEval < M > &__incr_eval,
           moMoveSelect < M > &__move_select, moSolContinue <EOT> &__continue, moComparator<EOT> &__acceptance_criterion,
           eoMonOp<EOT> &__perturbation, eoEvalFunc<EOT> &__full_eval):
        algo(*new moHC<M>(__move_init, __next_move, __incr_eval, __move_select, __full_eval)), cont(__continue),
        acceptance_criterion(__acceptance_criterion), perturbation(__perturbation), full_eval(__full_eval)
    {}

    //! Constructor for using a moTS for the moAlgo
    /*!
      \param __move_init The move initialisation (for the moTS).
      \param __next_move The move generator (for the moTS).
      \param __incr_eval The partial evaluation function (for the moTS).
      \param __tabu_list The tabu list (for the moTS !!!!).
      \param __aspir_crit The aspiration criterion (for the moTS).
      \param __moTS_continue The stopping criterion (for the moTS).
      \param __continue The stopping criterion.
      \param __acceptance_criterion The acceptance criterion.
      \param __perturbation The pertubation generator.
      \param __full_eval The evaluation function.
    */
    moILS (moMoveInit <M> &__move_init, moNextMove <M> &__next_move, moMoveIncrEval <M> &__incr_eval,
           moTabuList <M> &__tabu_list, moAspirCrit <M> &__aspir_crit, moSolContinue <EOT> &__moTS_continue,
           moSolContinue <EOT> &__continue, moComparator<EOT> &__acceptance_criterion, eoMonOp<EOT> &__perturbation,
           eoEvalFunc<EOT> &__full_eval):
        algo(*new moTS<M>(__move_init, __next_move, __incr_eval, __tabu_list, __aspir_crit, __moTS_continue, __full_eval)),
        cont(__continue), acceptance_criterion(__acceptance_criterion), perturbation(__perturbation), full_eval(__full_eval)
    {}

    //! Constructor for using a moSA for the moAlgo
    /*!
      \param __move_rand The random move generator (for the moSA).
      \param __incr_eval The partial evaluation function (for the moSA).
      \param __moSA_continue The stopping criterion (for the moSA).
      \param __init_temp The initial temperature (for the moSA).
      \param __cool_sched The cooling scheduler (for the moSA).
      \param __continue The stopping criterion.
      \param __acceptance_criterion The acceptance criterion.
      \param __perturbation The pertubation generator.
      \param __full_eval The evaluation function.
    */
    moILS (moRandMove<M> &__move_rand, moMoveIncrEval <M> &__incr_eval, moSolContinue <EOT> &__moSA_continue, double __init_temp,
           moCoolingSchedule & __cool_sched, moSolContinue <EOT> &__continue, moComparator<EOT> &__acceptance_criterion,
           eoMonOp<EOT> &__perturbation, eoEvalFunc<EOT> &__full_eval):
        algo(*new moSA<M>(__move_rand, __incr_eval, __moSA_continue, __init_temp, __cool_sched, __full_eval)),
        cont(__continue), acceptance_criterion(__acceptance_criterion), perturbation(__perturbation), full_eval(__full_eval)
    {}



    //! Function which launches the ILS
    /*!
       The ILS has to improve a current solution.
       As the moSA, the moTS and the moHC, it can be used for HYBRIDATION in an evolutionnary algorithm.

       \param __sol a current solution to improve.
       \return TRUE.
     */
    bool operator()(EOT & __sol)
    {
      EOT __sol_saved=__sol;

      cont.init ();

      //some code has been duplicated in order to avoid one perturbation and one evaluation without adding a test in the loop.
      // better than a do {} while; with a test in the loop.

      algo(__sol);

      if (acceptance_criterion(__sol, __sol_saved))
        {
          __sol_saved=__sol;

        }
      else
        {
          __sol=__sol_saved;
        }

      while (cont (__sol))
        {
          perturbation(__sol);
          full_eval(__sol);

          algo(__sol);

          if (acceptance_criterion(__sol, __sol_saved))
            {
              __sol_saved=__sol;
            }
          else
            {
              __sol=__sol_saved;
            }
        }

      return true;
    }

  private:

    //! The solution based heuristic.
    moAlgo<EOT> &algo;

    //! The stopping criterion.
    moSolContinue<EOT> &cont;

    //! The acceptance criterion.
    moComparator<EOT> &acceptance_criterion;

    //! The perturbation generator
    eoMonOp<EOT> &perturbation;

    //! The full evaluation function
    eoEvalFunc<EOT> &full_eval;
  };

#endif
