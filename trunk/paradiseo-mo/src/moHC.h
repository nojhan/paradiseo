/*
  <moHC.h>
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

#ifndef __moHC_h
#define __moHC_h

#include <eoEvalFunc.h>

#include <moAlgo.h>
#include <moHCMoveLoopExpl.h>

//! Hill Climbing (HC)
/*!
  Class which describes the algorithm for a hill climbing.
*/
template < class M >
class moHC:public moAlgo < typename M::EOType >
{
  //! Alias for the type.
  typedef typename M::EOType EOT;

  //! Alias for the fitness.
  typedef typename EOT::Fitness Fitness;

 public:
  
  //! Full constructor.
  /*!
    All the boxes are given in order the HC to use a moHCMoveLoopExpl.

    \param _move_initializer a move initialiser.
    \param _next_move_generator a neighborhood explorer.
    \param _incremental_evaluation a (generally) efficient evaluation function.
    \param _move_selection a move selector.
    \param _full_evaluation a full evaluation function.
  */
  moHC (moMoveInit < M > & _move_initializer, moNextMove < M > & _next_move_generator, 
	moMoveIncrEval < M > & _incremental_evaluation, moMoveSelect < M > & _move_selection, eoEvalFunc < EOT > & _full_evaluation) : 
  move_explorer(new moHCMoveLoopExpl<M>(_move_initializer, _next_move_generator, _incremental_evaluation, _move_selection)), 
    full_evaluation (_full_evaluation), move_explorer_memory_allocation(true)
  {}
  
  //! Light constructor.
  /*!
    This constructor allow to use another moMoveExpl (generally not a moHCMoveLoopExpl).

    \param _move_explorer a complete explorer.
    \param _full_evaluation a full evaluation function.
  */
  moHC (moMoveExpl < M > & _move_explorer, eoEvalFunc < EOT > & _full_evaluation): 
  move_explorer (_move_explorer), full_evaluation (_full_evaluation), move_explorer_memory_allocation(false)
  {}
  
  //! Destructor
  ~moHC()
    {
      if(move_explorer_memory_allocation)
	{
	  delete(move_explorer);
	}
    }

  //! Function which launches the HC
  /*!
    The HC has to improve a current solution.
    As the moSA and the mo TS, it can be used for HYBRIDATION in an evolutionnary algorithm.

    \param _solution a current solution to improve.
    \return true.
  */
  bool operator ()(EOT & _solution)
  {
    EOT new_solution;
    
    if ( _solution.invalid() )
      {
	full_evaluation(_solution);
      }
    
    new_solution=_solution;
    
    do
      {
	_solution=new_solution;
	(*move_explorer) (_solution, new_solution);
      }
    while ( new_solution.fitness() > _solution.fitness() );
    
    return true;
  }

 private:

  //! Complete exploration of the neighborhood.
  moMoveExpl < M > * move_explorer;

  //! A full evaluation function.
  eoEvalFunc < EOT > & full_evaluation;

  //! Indicate if the memory has been allocated for the move_explorer.
  bool move_explorer_memory_allocation;
};

#endif
