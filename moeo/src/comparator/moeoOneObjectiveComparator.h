/*
* <moeoOneObjectiveComparator.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
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
//-----------------------------------------------------------------------------

#ifndef MOEOONEOBJECTIVECOMPARATOR_H_
#define MOEOONEOBJECTIVECOMPARATOR_H_

#include <comparator/moeoComparator.h>

/**
 * Functor allowing to compare two solutions according to one objective.
 */
template < class MOEOT >
class moeoOneObjectiveComparator : public moeoComparator < MOEOT >
  {
  public:

    /**
     * Ctor.
     * @param _obj the index of objective
     */
    moeoOneObjectiveComparator(unsigned int _obj) : obj(_obj)
    {
      if (obj > MOEOT::ObjectiveVector::nObjectives())
        {
          throw eoException("Problem with the index of objective in moeoOneObjectiveComparator");
        }
    }


    /**
     * Returns true if _moeo1 < _moeo2 on the obj objective
     * @param _moeo1 the first solution
     * @param _moeo2 the second solution
     */
    bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
    {
      return _moeo1.objectiveVector()[obj] < _moeo2.objectiveVector()[obj];
    }


  private:

    /** the index of objective */
    unsigned int obj;

  };

#endif /*MOEOONEOBJECTIVECOMPARATOR_H_*/
