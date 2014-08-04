/*
* <moeoDistanceMatrix.h>
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

#ifndef MOEODISTANCEMATRIX_H_
#define MOEODISTANCEMATRIX_H_

#include <vector>
#include "../../eo/eoFunctor.h"
#include "moeoDistance.h"

/**
 * A matrix to compute distances between every pair of individuals contained in a population.
 */
template < class MOEOT , class Type >
class moeoDistanceMatrix : public eoUF < const eoPop < MOEOT > &, void > , public std::vector< std::vector < Type > >
  {
  public:

    using std::vector< std::vector < Type > > :: size;
    using std::vector< std::vector < Type > > :: operator[];


    /**
     * Ctor
     * @param _size size for every dimension of the matrix
     * @param _distance the distance to use
     */
    moeoDistanceMatrix (unsigned int _size, moeoDistance < MOEOT , Type > & _distance) : distance(_distance)
    {
      this->resize(_size);
      for (unsigned int i=0; i<_size; i++)
        {
          this->operator[](i).resize(_size);
        }
    }


    /**
     * Sets the distance between every pair of individuals contained in the population _pop
     * @param _pop the population
     */
    void operator()(const eoPop < MOEOT > & _pop)
    {
      // 1 - setup the bounds (if necessary)
      distance.setup(_pop);
      // 2 - compute distances
      this->operator[](0).operator[](0) = Type();
      for (unsigned int i=0; i<size(); i++)
        {
          this->operator[](i).operator[](i) = Type();
          for (unsigned int j=0; j<i; j++)
            {
              this->operator[](i).operator[](j) = distance(_pop[i], _pop[j]);
              this->operator[](j).operator[](i) = this->operator[](i).operator[](j);
            }
        }
    }


  private:

    /** the distance to use */
    moeoDistance < MOEOT , Type > & distance;

  };

#endif /*MOEODISTANCEMATRIX_H_*/
