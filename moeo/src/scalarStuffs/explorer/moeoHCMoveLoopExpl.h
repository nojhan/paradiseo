/*
   <moeoHCMoveLoopExpl.h>
   Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
   (C) OPAC Team, LIFL, 2002-2008

   Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)

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

#ifndef __moeoHCLoopExpl_h
#define __moeoHCLoopExpl_h

#include <moMoveLoopExpl.h>
#include <moMoveInit.h>
#include <moNextMove.h>
#include <moMoveIncrEval.h>
#include <moMoveSelect.h>
#include <moeo>
#include <scalarStuffs/fitness/moeoIncrEvalSingleObjectivizer.h>
/**
  move explorer for multiobjectives solutions
 */
template < class M >
class moeoHCMoveLoopExpl:public moMoveLoopExpl < M >
{
	//! Alias for the type.
	typedef typename M::EOType EOT;

	//! Alias for the fitness.
	typedef typename M::EOType::Fitness Fitness;

	typedef typename M::EOType::ObjectiveVector ObjectiveVector;

	public:

	//! Constructor.
	/*!
	  All the boxes have to be specified.

	  \param _move_initializer The move initialiser.
	  \param _next_move_generator The neighbourhood explorer.
	  \param _incremental_evaluation (generally) Efficient evaluation function.
	  \param _move_selection The move selector.
	 */
	moeoHCMoveLoopExpl (moMoveInit < M > & _move_initializer, moNextMove < M > & _next_move_generator, 
			moeoIncrEvalSingleObjectivizer < EOT,M > & _incremental_evaluation, moMoveSelect < M > & _move_selection) :
		move_initializer (_move_initializer), next_move_generator (_next_move_generator), 
		incremental_evaluation (_incremental_evaluation), move_selection (_move_selection)
	{}

	//!  Procedure which launches the explorer.
	/*!
	  The exploration starts from an old solution and provides a new solution.

	  \param _old_solution The current solution.
	  \param _new_solution The new solution (result of the procedure).
	 */
	void operator () (const EOT & _old_solution, EOT & _new_solution)
	{
		M move, best_move;
		Fitness best_fitness;
		bool has_next_move, selection_update_is_ok;

		if( _old_solution.invalid() )
		{
			throw std::runtime_error("[moHCMoveLoopExpl.h]: The current solution has not been evaluated.");
		}

		/*
		   The two following lines are added to avoid compilation warning.
		   <=> current best move fitness is the current fitness.
		   <=> move and best move are empty for the moment.
		 */
		best_fitness=_old_solution.fitness();
		move=best_move;
		//At the begining, the new sol is equivalent to the old one.
		_new_solution=(EOT)_old_solution;

		// Restarting the exploration of the neighbourhood
		move_initializer(move, _old_solution); 

		move_selection.init(_old_solution.fitness ());

		do
		{
			selection_update_is_ok = move_selection.update (move, incremental_evaluation(move, _old_solution) );
			has_next_move = next_move_generator (move, _old_solution);
		}
		while ( selection_update_is_ok && has_next_move);
		//The selecter gives the value of the best move and the corresponding best fitness.
		move_selection (best_move, best_fitness);

		/*std::cout<<"bonjour moeoloopexpl"<<std::endl;   
		  for (unsigned i=0;i<6;i++){
		  std::cout<<"move"<<best_move[i]<<std::endl;
		  } */
		//The best move is applied on the new solution.
		best_move(_new_solution);

		//fitness and objective are set. 
		_new_solution.fitness(best_fitness);
		_new_solution.objectiveVector(incremental_evaluation.incr_obj(best_move, _old_solution));
		//we make a full eval
		//    incremental_evaluation(_new_solution);
	}

	private:

	//! Move initialiser.
	moMoveInit < M > & move_initializer;

	//! Neighborhood explorer.
	moNextMove < M > & next_move_generator;

	//! (generally) Efficient evaluation.
	moeoIncrEvalSingleObjectivizer < EOT,M > & incremental_evaluation;

	//! Move selector.
	moMoveSelect < M > & move_selection;
};

#endif
