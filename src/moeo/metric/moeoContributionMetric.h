/*
* <moeoContributionMetric.h>
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

#ifndef MOEOCONTRIBUTIONMETRIC_H_
#define MOEOCONTRIBUTIONMETRIC_H_

#include "../comparator/moeoParetoObjectiveVectorComparator.h"
#include "moeoMetric.h"

/**
 * The contribution metric evaluates the proportion of non-dominated solutions given by a Pareto set relatively to another Pareto set
 * (Meunier, Talbi, Reininger: 'A multiobjective genetic algorithm for radio network optimization', in Proc. of the 2000 Congress on Evolutionary Computation, IEEE Press, pp. 317-324)
 */
template < class ObjectiveVector >
class moeoContributionMetric : public moeoVectorVsVectorBinaryMetric < ObjectiveVector, double >
  {
  public:

    /**
     * Returns the contribution of the Pareto set '_set1' relatively to the Pareto set '_set2'
     * @param _set1 the first Pareto set
     * @param _set2 the second Pareto set
     */
    double operator()(const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
    {
      unsigned int c  = card_C(_set1, _set2);
      unsigned int w1 = card_W(_set1, _set2);
      unsigned int n1 = card_N(_set1, _set2);
      unsigned int w2 = card_W(_set2, _set1);
      unsigned int n2 = card_N(_set2, _set1);
      return (double) (c / 2.0 + w1 + n1) / (c + w1 + n1 + w2 + n2);
    }


  private:

    /** Functor to compare two objective vectors according to Pareto dominance relation */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;


    /**
     * Returns the number of solutions both in '_set1' and '_set2'
     * @param _set1 the first Pareto set
     * @param _set2 the second Pareto set
     */
    unsigned int card_C (const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
    {
      unsigned int c=0;
      for (unsigned int i=0; i<_set1.size(); i++)
        for (unsigned int j=0; j<_set2.size(); j++)
          if (_set1[i] == _set2[j])
            {
              c++;
              break;
            }
      return c;
    }


    /**
     * Returns the number of solutions in '_set1' dominating at least one solution of '_set2'
     * @param _set1 the first Pareto set
     * @param _set2 the second Pareto set
     */
    unsigned int card_W (const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
    {
      unsigned int w=0;
      for (unsigned int i=0; i<_set1.size(); i++)
        for (unsigned int j=0; j<_set2.size(); j++)
          if (paretoComparator(_set2[j], _set1[i]))
            {
              w++;
              break;
            }
      return w;
    }


    /**
     * Returns the number of solutions in '_set1' having no relation of dominance with those from '_set2'
     * @param _set1 the first Pareto set
     * @param _set2 the second Pareto set
     */
    unsigned int card_N (const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
    {
      unsigned int n=0;
      for (unsigned int i=0; i<_set1.size(); i++)
        {
          bool domin_rel = false;
          for (unsigned int j=0; j<_set2.size(); j++)
            if ( (paretoComparator(_set2[j], _set1[i])) || (paretoComparator(_set1[i], _set2[j])) )
              {
                domin_rel = true;
                break;
              }
          if (! domin_rel)
            n++;
        }
      return n;
    }

  };

#endif /*MOEOCONTRIBUTIONMETRIC_H_*/
