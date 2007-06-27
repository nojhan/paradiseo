// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moExponentialCoolingSchedule.h"

// (c) OPAC Team, LIFL, 2003-2007

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moExponentialCoolingSchedule_h
#define __moExponentialCoolingSchedule_h

#include "moCoolingSchedule.h"

//! One of the possible moCoolingSchedule
/*!
  An other very simple cooling schedule, the temperature decrease according to a ratio while
  the temperature is greater than a given threshold.
 */
class moExponentialCoolingSchedule: public moCoolingSchedule
{

public:
  //! Simple constructor
  /*!
     \param __threshold the threshold.
     \param __ratio the ratio used to descrease the temperature.
   */
  moExponentialCoolingSchedule (double __threshold, double __ratio):threshold (__threshold), ratio (__ratio)
  {}

  //! Function which proceeds to the cooling.
  /*!
     It decreases the temperature and indicates if it is greater than the threshold.

     \param __temp the current temperature.
     \return if the new temperature (current temperature * ratio) is greater than the threshold.
   */
  bool operator() (double &__temp)
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
