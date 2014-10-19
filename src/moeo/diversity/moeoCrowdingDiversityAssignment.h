/*
* <moeoCrowdingDiversityAssignment.h>
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

#ifndef MOEOCROWDINGDIVERSITYASSIGNMENT_H_
#define MOEOCROWDINGDIVERSITYASSIGNMENT_H_

#include "../../eo/eoPop.h"
#include "../comparator/moeoOneObjectiveComparator.h"
#include "moeoDiversityAssignment.h"

/**
 * Diversity assignment sheme based on crowding proposed in:
 * K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation, vol. 6, no. 2 (2002).
 */
template < class MOEOT >
class moeoCrowdingDiversityAssignment : public moeoDiversityAssignment < MOEOT >
  {
  public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Returns a big value (regarded as infinite)
     */
    double inf() const
      {
        return std::numeric_limits<double>::max();
      }


    /**
     * Returns a very small value that can be used to avoid extreme cases (where the min bound == the max bound)
     */
    double tiny() const
      {
        return 1e-6;
      }


    /**
     * Computes diversity values for every solution contained in the population _pop
     * @param _pop the population
     */
    void operator()(eoPop < MOEOT > & _pop)
    {
      if (_pop.size() <= 2)
        {
          for (unsigned int i=0; i<_pop.size(); i++)
            {
              _pop[i].diversity(inf());
            }
        }
      else
        {
          setDistances(_pop);
        }
    }


    /**
     * @warning NOT IMPLEMENTED, DO NOTHING !
     * Updates the diversity values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     * @warning NOT IMPLEMENTED, DO NOTHING !
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
      std::cout << "WARNING : updateByDeleting not implemented in moeoCrowdingDiversityAssignment" << std::endl;
    }


  protected:

    /**
     * Sets the distance values
     * @param _pop the population
     */
    virtual void setDistances (eoPop < MOEOT > & _pop)
    {
      double min, max, distance;
      unsigned int nObjectives = MOEOT::ObjectiveVector::nObjectives();
      // set diversity to 0
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          _pop[i].diversity(0.0);
        }
      // for each objective
      for (unsigned int obj=0; obj<nObjectives; obj++)
        {
          // comparator
          moeoOneObjectiveComparator < MOEOT > objComp(obj);
          // sort
          std::sort(_pop.begin(), _pop.end(), objComp);
          // min & max
          min = _pop[0].objectiveVector()[obj];
          max = _pop[_pop.size()-1].objectiveVector()[obj];
          // set the diversity value to infiny for min and max
          _pop[0].diversity(inf());
          _pop[_pop.size()-1].diversity(inf());
          for (unsigned int i=1; i<_pop.size()-1; i++)
            {
              distance = (_pop[i+1].objectiveVector()[obj] - _pop[i-1].objectiveVector()[obj]) / (max-min);
              _pop[i].diversity(_pop[i].diversity() + distance);
            }
        }
    }

  };

#endif /*MOEOCROWDINGDIVERSITYASSIGNMENT_H_*/
