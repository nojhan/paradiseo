/*
* <moeoAchievementFitnessAssignment.h>
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

#ifndef MOEOACHIEVEMENTFITNESSASSIGNMENT_H_
#define MOEOACHIEVEMENTFITNESSASSIGNMENT_H_

#include <vector>
#include "../../../eo/eoPop.h"
#include "../../fitness/moeoScalarFitnessAssignment.h"

/**
 * Fitness assignment sheme based on the achievement scalarizing function propozed by Wiersbicki (1980).
 */
template < class MOEOT >
class moeoAchievementFitnessAssignment : public moeoScalarFitnessAssignment < MOEOT >
  {
  public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Default ctor
     * @param _reference reference point vector
     * @param _lambdas weighted coefficients vector
     * @param _spn arbitrary small positive number (0 < _spn << 1)
     */
    moeoAchievementFitnessAssignment(ObjectiveVector & _reference, std::vector < double > & _lambdas, double _spn=0.0001) : reference(_reference), lambdas(_lambdas), spn(_spn)
    {
      // consistency check
      if ((spn < 0.0) || (spn > 1.0))
        {
          std::cout << "Warning, the arbitrary small positive number should be > 0 and <<1, adjusted to 0.0001\n";
          spn = 0.0001;
        }
    }


    /**
     * Ctor with default values for lambdas (1/nObjectives)
     * @param _reference reference point vector
     * @param _spn arbitrary small positive number (0 < _spn << 1)
     */
    moeoAchievementFitnessAssignment(ObjectiveVector & _reference, double _spn=0.0001) : reference(_reference), spn(_spn)
    {
      // compute the default values for lambdas
      lambdas  = std::vector < double > (ObjectiveVector::nObjectives());
      for (unsigned int i=0 ; i<lambdas.size(); i++)
        {
          lambdas[i] = 1.0 / ObjectiveVector::nObjectives();
        }
      // consistency check
      if ((spn < 0.0) || (spn > 1.0))
        {
          std::cout << "Warning, the arbitrary small positive number should be > 0 and <<1, adjusted to 0.0001\n";
          spn = 0.0001;
        }
    }


    /**
     * Sets the fitness values for every solution contained in the population _pop
     * @param _pop the population
     */
    virtual void operator()(eoPop < MOEOT > & _pop)
    {
      for (unsigned int i=0; i<_pop.size() ; i++)
        {
          compute(_pop[i]);
        }
    }


    /**
     * Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account (nothing to do).
     * @param _pop the population
     * @param _objVec the objective vector
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
      // nothing to do ;-)
    }


    /**
     * Sets the reference point
     * @param _reference the new reference point
     */
    void setReference(const ObjectiveVector & _reference)
    {
      reference = _reference;
    }


  private:

    /** the reference point */
    ObjectiveVector reference;
    /** the weighted coefficients vector */
    std::vector < double > lambdas;
    /** an arbitrary small positive number (0 < _spn << 1) */
    double spn;


    /**
     * Returns a big value (regarded as infinite)
     */
    double inf() const
      {
        return std::numeric_limits<double>::max();
      }


    /**
     * Computes the fitness value for a solution
     * @param _moeo the solution
     */
    void compute(MOEOT & _moeo)
    {
      unsigned int nobj = MOEOT::ObjectiveVector::nObjectives();
      double temp;
      double min = inf();
      double sum = 0;
      for (unsigned int obj=0; obj<nobj; obj++)
        {
          temp = lambdas[obj] * (reference[obj] - _moeo.objectiveVector()[obj]);
          min = std::min(min, temp);
          sum += temp;
        }
      _moeo.fitness(min + spn*sum);
    }

  };

#endif /*MOEOACHIEVEMENTFITNESSASSIGNMENT_H_*/
