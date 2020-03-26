/*
  <moeoTSMoveLoopExpl.h>
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

#ifndef _moeoTSMoveLoopExpl_h
#define _moeoTSMoveLoopExpl_h
#include <oldmo>
#include <moMoveLoopExpl.h>
#include <moMoveInit.h>
#include <moNextMove.h>
#include <moMoveIncrEval.h>
#include <moMoveSelect.h>
#include <moTabuList.h>
#include <moAspirCrit.h>
#include <moBestImprSelect.h>



template <class M>
class moeoTSMoveLoopExpl:public moMoveLoopExpl < M >
{
	//!Alias for the type
	typedef typename M::EOType EOT;

	//!Alias for the fitness
	typedef typename M::EOType::Fitness Fitness;

	public:

	//!Constructor
	/*!
	  \param _move_initializer The move initializer.
	  \param _next_move_generator The neighbourhood explorer.
	  \param _incremental_evaluation A (generally) efficient evaluation.
	  \param _tabu_list The tabu list.
	  \param _aspiration_criterion An aspiration criterion.
	 */
	moeoTSMoveLoopExpl (moMoveInit < M > & _move_initializer, moNextMove < M > & _next_move_generator, moeoIncrEvalSingleObjectivizer < EOT, M > & _incremental_evaluation, moTabuList < M > & _tabu_list,  moAspirCrit < M > & _aspiration_criterion):
		move_initializer(_move_initializer), 
		next_move_generator(_next_move_generator), 
		incremental_evaluation(_incremental_evaluation),
		tabu_list(_tabu_list), 
		aspiration_criterion(_aspiration_criterion)
	{
		tabu_list.init ();
		aspiration_criterion.init ();
	}
	//!Procedure which lauches the exploration
	/*!
	  The exploration continues while the chosen move is not in the tabu list 
	  or the aspiration criterion is true. If these 2 conditions are not true, the
	  exploration stops if the move selector update function returns false.

	  \param _old_solution the initial solution
	  \param _new_solution the new solution
	 */
	void operator () (const EOT & _old_solution, EOT & _new_solution)
	{
		M move, best_move;
		Fitness fitness, best_move_fitness;

    bool move_is_tabu, aspiration_criterion_is_verified, selection_update_is_ok, has_next_move;

    if( _old_solution.invalidFitness() || _old_solution.invalid() )
      {
	throw eoInvalidFitnessError("[moTSMoveLoopExpl.h]: The current solution has not been evaluated.");
      }
    
    //At the begining, the new solution is equivalent to the old one.
    _new_solution=(EOT)_old_solution;
 //   EOT mem(_old_solution);

 
    // Restarting the exploration of  of the neighborhood !
    move_initializer (move, _old_solution);	

    move_selection.init( _old_solution.fitness() );

    selection_update_is_ok=true;
//    std::cout<<"moeoTS lets go"<<std::cout;
    do
      {
	fitness = incremental_evaluation(move, _old_solution);
//	std::cout<<"fit: "<<fitness<<std::endl;

	move_is_tabu = tabu_list(move, _old_solution);

	aspiration_criterion_is_verified = aspiration_criterion(move, fitness);

	if( !move_is_tabu || aspiration_criterion_is_verified )
	  {
	    selection_update_is_ok = move_selection.update(move, fitness);
	  }

	has_next_move = next_move_generator(move, _old_solution);
      }
    while( has_next_move && selection_update_is_ok );
  //  std::cout<<"moeoTS before select"<<std::cout;

    move_selection(best_move, best_move_fitness);
    typename EOT::ObjectiveVector best_obj=incremental_evaluation.incr_obj(best_move,_new_solution);

    //std::cout<<"moeo explo apply move "<<std::endl;
    // Apply the best move.
    best_move(_new_solution);
    
    // The fitness is set to avoid an additionnal fitness computation.
   // std::cout<<"moeo explo apply fit"<<std::endl;
    _new_solution.fitness(best_move_fitness);
   // std::cout<<"moeo explo apply obj"<<std::endl;
    _new_solution.objectiveVector(best_obj);
   // std::cout<<"moeo explo apply obj OK"<<std::endl;
  //  incremental_evaluation(_new_solution);
      
    // Removing moves that are no more tabu.
    tabu_list.update ();
    
    // Updating the tabu list
    tabu_list.add(best_move, _new_solution);
    //std::cout<<"moeo end "<<std::endl;
  }

 private:

  //! Move initialisation
  moMoveInit < M > & move_initializer;

  //! Neighborhood explorer
  moNextMove < M > & next_move_generator;

  //! Efficient evaluation
  moeoIncrEvalSingleObjectivizer < EOT,M > & incremental_evaluation;

  //! Move selector
  moBestImprSelect < M > move_selection;

  //! Tabu list
  moTabuList < M > & tabu_list;

  //! Aspiration criterion
  moAspirCrit < M > & aspiration_criterion;
};

#endif
