// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moImprAspirCrit.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moImprBestFitAspirCrit_h
#define __moImprBestFitAspirCrit_h

#include "moAspirCrit.h"

//! One of the possible moAspirCrit
/*!
  This criterion is satisfied when a given fitness
  is the best ever considered.
*/
template < class M > class moImprBestFitAspirCrit:public moAspirCrit < M >
{

public:

  //! Alias for the fitness  
  typedef typename M::EOType::Fitness Fitness;

  //! Contructor
  moImprBestFitAspirCrit ()
  {

    first_time = true;
  }

  //! Initialisation procedure
  void init ()
  {

    first_time = true;
  }

  //! Function that indicates if the fit is better that the already saved fit
  /*!
     The first time, the function only saved the current move and fitness.

     \param __move a move.
     \param __fit a fitnes linked to the move.
     \return TRUE the first time and if __fit > best_fit, else FALSE.
   */
  bool operator   () (const M & __move, const Fitness & __fit)
  {

    if (first_time)
      {

	best_fit = __fit;
	first_time = false;

	return true;
      }
    else if (__fit < best_fit)
      return false;

    else
      {

	best_fit = __fit;

	return true;
      }
  }

private:

  //! Best fitness found until now
  Fitness best_fit;

  //! Indicates that a fitness has been already saved or not
  bool first_time;
};

#endif
