/*
* <moeoObjectiveVectorTraits.h>
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

#ifndef MOEOOBJECTIVEVECTORTRAITS_H_
#define MOEOOBJECTIVEVECTORTRAITS_H_

#include <iostream>
#include <stdexcept>
#include <vector>

/**
 * A traits class for moeoObjectiveVector to specify the number of objectives and which ones have to be minimized or maximized.
 */
class moeoObjectiveVectorTraits
  {
  public:

    /**
     * Parameters setting
     * @param _nObjectives the number of objectives
     * @param _bObjectives the min/max vector (true = min / false = max)
     */
    static void setup(unsigned int _nObjectives, std::vector < bool > & _bObjectives)
    {
      // in case the number of objectives was already set to a different value
      if ( nObj && (nObj != _nObjectives) )
        {
          std::cout << "WARNING\n";
          std::cout << "WARNING : the number of objectives are changing\n";
          std::cout << "WARNING : Make sure all existing objects are destroyed\n";
          std::cout << "WARNING\n";
        }
      // number of objectives
      nObj = _nObjectives;
      // min/max vector
      bObj = _bObjectives;
      // in case the number of objectives and the min/max vector size don't match
      if (nObj != bObj.size())
        throw std::runtime_error("Number of objectives and min/max size don't match in moeoObjectiveVectorTraits::setup");
    }


    /**
     * Returns the number of objectives
     */
    static unsigned int nObjectives()
    {
      // in case the number of objectives would not be assigned yet
      if (! nObj)
        throw std::runtime_error("Number of objectives not assigned in moeoObjectiveVectorTraits");
      return nObj;
    }


    /**
     * Returns true if the _ith objective have to be minimized
     * @param _i  the index
     */
    static bool minimizing(unsigned int _i)
    {
      // in case there would be a wrong index
      if (_i >= bObj.size())
        throw std::runtime_error("Wrong index in moeoObjectiveVectorTraits");
      return bObj[_i];
    }


    /**
     * Returns true if the _ith objective have to be maximized
     * @param _i  the index
     */
    static bool maximizing(unsigned int _i)
    {
      return (! minimizing(_i));
    }


    /**
     * Returns the tolerance value (to compare solutions)
     */
    static double tolerance()
    {
      return 1e-10;
    }


  private:

    /** The number of objectives */
    static unsigned int nObj;
    /** The min/max vector */
    static std::vector < bool > bObj;

  };

#endif /*MOEOOBJECTIVEVECTORTRAITS_H_*/
