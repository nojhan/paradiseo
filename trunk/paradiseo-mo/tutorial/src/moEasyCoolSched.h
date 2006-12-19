// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moEasyCoolSched.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moEasyCoolSched_h
#define __moEasyCoolSched_h

#include "moCoolSched.h"

//! One of the possible moCoolSched
/*!
  The simpliest, the temperature decrease according to a ratio until
  it greater than a threshold.
 */
class moEasyCoolSched:public moCoolSched
{

public:
  //! Simple constructor
  /*!
     \param __threshold the threshold.
     \param __ratio the ratio used to descrease the temperature.
   */
  moEasyCoolSched (double __threshold,
		   double __ratio):threshold (__threshold), ratio (__ratio)
  {

  }

  //! Function which proceeds to the cooling.
  /*!
     Decrease the temperature and indicates if it is greater than the threshold.

     \param __temp the current temperature.
     \return if the new temperature (current temperature * ratio) is greater than the threshold.
   */
  bool operator   () (double &__temp)
  {

    return (__temp *= ratio) > threshold;
  }

private:

  //! The temperature threhold.
  double threshold;

  //! The decreasing factor of the temperature.
  double ratio;

};

#endif
