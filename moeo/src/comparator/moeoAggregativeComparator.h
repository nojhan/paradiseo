/*
* <moeoAggregativeComparator.h>
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

#ifndef MOEOAGGREGATIVECOMPARATOR_H_
#define MOEOAGGREGATIVECOMPARATOR_H_

#include <comparator/moeoComparator.h>

/**
 * Functor allowing to compare two solutions according to their fitness and diversity values, each according to its aggregative value.
 */
template < class MOEOT >
class moeoAggregativeComparator : public moeoComparator < MOEOT >
  {
  public:

    /**
     * Ctor.
     * @param _weightFitness the weight for fitness
     * @param _weightDiversity the weight for diversity
     */
    moeoAggregativeComparator(double _weightFitness = 1.0, double _weightDiversity = 1.0) : weightFitness(_weightFitness), weightDiversity(_weightDiversity)
    {}


    /**
     * Returns true if _moeo1 < _moeo2 according to the aggregation of their fitness and diversity values
     * @param _moeo1 the first solution
     * @param _moeo2 the second solution
     */
    bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
    {
      return ( weightFitness * _moeo1.fitness() + weightDiversity * _moeo1.diversity() ) < ( weightFitness * _moeo2.fitness() + weightDiversity * _moeo2.diversity() );
    }


  private:

    /** the weight for fitness */
    double weightFitness;
    /** the weight for diversity */
    double weightDiversity;

  };

#endif /*MOEOAGGREGATIVECOMPARATOR_H_*/
