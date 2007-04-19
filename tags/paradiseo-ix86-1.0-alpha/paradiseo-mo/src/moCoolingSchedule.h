// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moCoolingSchedule.h"

// (c) OPAC Team, LIFL, 2003-2007

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moCoolingSchedule_h
#define __moCoolingSchedule_h

#include <eoFunctor.h>

//! This class gives the description of a cooling schedule.
/*!
  It is only a description... An object that herits from this class is needed to be used in a moSA.
  See moExponentialCoolingSchedule or moLinearCoolingSchedule for example.
*/
class moCoolingSchedule:public eoUF < double &, bool >
{

};

#endif
