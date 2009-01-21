/*
  <moImprBestFitAspirCrit.h>
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

#ifndef _moImprBestFitAspirCrit_h
#define _moImprBestFitAspirCrit_h

#include <moAspirCrit.h>

//! One of the possible moAspirCrit
/*!
  This criterion is satisfied when a given fitness
  is the best ever considered.
*/
template < class M >
class moImprBestFitAspirCrit:public moAspirCrit < M >
{

 public:

  //! Alias for the fitness
  typedef typename M::EOType::Fitness Fitness;

  //! Contructor
  moImprBestFitAspirCrit (): first_time(true)
    {}

  //! Initialisation procedure
  void init ()
  {
    first_time = true;
  }

  //! Function that indicates if the current fitness is better that the already saved fitness
  /*!
    The first time, the function only saved the current move and fitness.

    \param _move A move.
    \param _fitness A fitness linked to the move.
    \return true The first time and if _fitness > best_fitness, else false.
  */
  bool operator () (const M & _move, const Fitness & _fitness)
  {
    //code only used for avoiding warning because _move is not used in this function.
    const M move(_move);

    if (first_time)
      {
	best_fitness = _fitness;
	first_time = false;

	return true;
      }

    if (_fitness > best_fitness)
      {
    	best_fitness = _fitness;
    	return true;
      }   
    return false;
  }

 private:

  //! Best fitness found until now
  Fitness best_fitness;

  //! Indicates that a fitness has been already saved or not
  bool first_time;
};

#endif
