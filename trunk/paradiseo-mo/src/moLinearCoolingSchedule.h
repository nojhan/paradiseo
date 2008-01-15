/*
  <moLinearCoolingSchedule.h>
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

#ifndef _moLinearCoolingSchedule_h
#define _moLinearCoolingSchedule_h

#include <moCoolingSchedule.h>

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
    \param _threshold the threshold.
    \param _quantity the quantity used to descrease the temperature.
  */
  moLinearCoolingSchedule (double _threshold, double _quantity):threshold (_threshold), quantity (_quantity)
  {}

  //! Function which proceeds to the cooling.
  /*!
    It decreases the temperature and indicates if it is greater than the threshold.

    \param _current_temperature The current temperature.
    \return true if the new temperature (current temperature - quantity) is greater than the threshold, false otherwise.
  */
  bool operator() (double & _current_temperature)
  {
    return (_current_temperature -= quantity) > threshold;
  }

 private:

  //! The temperature threhold.
  double threshold;

  //! The quantity that allows the temperature to decrease.
  double quantity;
};

#endif
