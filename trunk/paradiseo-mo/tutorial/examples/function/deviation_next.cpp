/*
* <deviation_next.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Sébastien Cahon, Jean-Charles Boisson
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#include "deviation_next.h"

DeviationNext::DeviationNext(double _bound, double _step): bound(_bound), step(_step)
{
  if(bound<0.0)
    {
      std::cout << "[affectation_next.cpp][DeviationNext]: bound is negative, " << bound << " is tranformed to ";
      bound=-bound;
      std::cout << bound << "." << std::endl;
    }

  if(step<0.0)
    {
      std::cout << "[affectation_next.cpp][DeviationNext]: step is negative, " << step << " is tranformed to ";
      step=-step;
      std::cout << step << "." << std::endl;
    }

  if(step>bound)
    {
      std::cout << "[affectation_next.cpp][DeviationNext]: step is higher than bound, " << step << " is tranformed to ";
      step = bound / 2;
      std::cout << step << "." << std::endl;
    }
}

bool DeviationNext::operator () (Deviation & _move, const Affectation & _affectation)
{
  Affectation affectation(_affectation);

  double deltaX1, deltaX2;

  deltaX1=_move.first;
  deltaX2=_move.second;

  //std::cout << "deltaX1 = " << deltaX1 << ", deltaX2 = " << deltaX2 << std::endl;

  if( (deltaX1>=bound) && (deltaX2)>=bound )
    {
      return false;
    }

  if(deltaX2 >= bound)
    {
      deltaX1+=step;
      deltaX2=-bound;
      
      _move.first=deltaX1;
      _move.second=deltaX2;

      return true;
    }

  deltaX2+=step;
  
  _move.second=deltaX2;

  return true;
}
