/*
* <moeoObjectiveVector.h>
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

#ifndef MOEOOBJECTIVEVECTOR_H_
#define MOEOOBJECTIVEVECTOR_H_

#include <vector>

/**
 * Abstract class allowing to represent a solution in the objective space (phenotypic representation).
 * The template argument ObjectiveVectorTraits defaults to moeoObjectiveVectorTraits,
 * but it can be replaced at will by any other class that implements the static functions defined therein.
 * Some static funtions to access to the traits characteristics are re-defined in order not to write a lot of typedef's.
 */
template < class ObjectiveVectorTraits, class ObjectiveVectorType >
class moeoObjectiveVector : public std::vector < ObjectiveVectorType >
  {
  public:

    /** The traits of objective vectors */
    typedef ObjectiveVectorTraits Traits;
    /** The type of an objective value */
    typedef ObjectiveVectorType Type;


    /**
     * Ctor
     */
    moeoObjectiveVector(Type _value = Type()) : std::vector < Type > (ObjectiveVectorTraits::nObjectives(), _value)
    {}


    /**
     * Ctor from a vector of Type
     * @param _v the std::vector < Type >
     */
    moeoObjectiveVector(std::vector < Type > & _v) : std::vector < Type > (_v)
    {}


    /**
     * Parameters setting (for the objective vector of any solution)
     * @param _nObjectives the number of objectives
     * @param _bObjectives the min/max vector (true = min / false = max)
     */
    static void setup(unsigned int _nObjectives, std::vector < bool > & _bObjectives)
    {
      ObjectiveVectorTraits::setup(_nObjectives, _bObjectives);
    }


    /**
     * Returns the number of objectives
     */
    static unsigned int nObjectives()
    {
      return ObjectiveVectorTraits::nObjectives();
    }


    /**
     * Returns true if the _ith objective have to be minimized
     * @param _i  the index
     */
    static bool minimizing(unsigned int _i)
    {
      return ObjectiveVectorTraits::minimizing(_i);
    }


    /**
     * Returns true if the _ith objective have to be maximized
     * @param _i  the index
     */
    static bool maximizing(unsigned int _i)
    {
      return ObjectiveVectorTraits::maximizing(_i);
    }

  };

#endif /*MOEOOBJECTIVEVECTOR_H_*/
