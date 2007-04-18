// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moLinearCoolingSchedule.h"

// (c) OPAC Team, LIFL, 2003-2007

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moLinearCoolingSchedule_h
#define __moLinearCoolingSchedule_h

#include "moCoolingSchedule.h"

//! One of the possible moCoolingSchedule
/*!
  An another very simple cooling schedule, the temperature decrease according to a quantity while
  the temperature is greater than a threshold.
 */
class moLinearCoolingSchedule: public moCoolingSchedule
{

public:
  //! Simple constructor
  /*!
     \param __threshold the threshold.
     \param __quantity the quantity used to descrease the temperature.
   */
  moLinearCoolingSchedule (double __threshold, double __quantity):threshold (__threshold), quantity (__quantity)
  {}

  //! Function which proceeds to the cooling.
  /*!
     It decreases the temperature and indicates if it is greater than the threshold.

     \param __temp the current temperature.
     \return if the new temperature (current temperature - quantity) is greater than the threshold.
   */
  bool operator() (double &__temp)
  {
    return (__temp -= quantity) > threshold;
  }

private:

  //! The temperature threhold.
  double threshold;

  //! The quantity that allows the temperature to decrease.
  double quantity;
};

#endif
