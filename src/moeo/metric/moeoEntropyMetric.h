/*
* <moeoEntropyMetric.h>
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

#ifndef MOEOENTROPYMETRIC_H_
#define MOEOENTROPYMETRIC_H_

#include <vector>
#include "../comparator/moeoParetoObjectiveVectorComparator.h"
#include "moeoMetric.h"

/**
 * The entropy gives an idea of the diversity of a Pareto set relatively to another
 * (Basseur, Seynhaeve, Talbi: 'Design of Multi-objective Evolutionary Algorithms: Application to the Flow-shop Scheduling Problem', in Proc. of the 2002 Congress on Evolutionary Computation, IEEE Press, pp. 1155-1156)
 */
template < class ObjectiveVector >
class moeoEntropyMetric : public moeoVectorVsVectorBinaryMetric < ObjectiveVector, double >
  {
  public:

    /**
     * Returns the entropy of the Pareto set '_set1' relatively to the Pareto set '_set2'
     * @param _set1 the first Pareto set
     * @param _set2 the second Pareto set
     */
    double operator()(const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
    {
      // normalization
      std::vector< ObjectiveVector > set1 = _set1;
      std::vector< ObjectiveVector > set2= _set2;
      removeDominated (set1);
      removeDominated (set2);
      prenormalize (set1);
      normalize (set1);
      normalize (set2);

      // making of PO*
      std::vector< ObjectiveVector > star; // rotf :-)
      computeUnion (set1, set2, star);
      removeDominated (star);

      // making of PO1 U PO*
      std::vector< ObjectiveVector > union_set1_star; // rotf again ...
      computeUnion (set1, star, union_set1_star);

      unsigned int C = union_set1_star.size();
      float omega=0;
      float entropy=0;

      for (unsigned int i=0 ; i<C ; i++)
        {
          unsigned int N_i = howManyInNicheOf (union_set1_star, union_set1_star[i], star.size());
          unsigned int n_i = howManyInNicheOf (set1, union_set1_star[i], star.size());
          if (n_i > 0)
            {
              omega += 1.0 / N_i;
              entropy += (float) n_i / (N_i * C) * log (((float) n_i / C) / log (2.0));
            }
        }
      entropy /= - log (omega);
      entropy *= log (2.0);
      return entropy;
    }


  private:

    /** vector of min values */
    std::vector<double> vect_min_val;
    /** vector of max values */
    std::vector<double> vect_max_val;
    /** Functor to compare two objective vectors according to Pareto dominance relation */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;


    /**
     * Removes the dominated individuals contained in _f
     * @param _f a Pareto set
     */
    void removeDominated(std::vector < ObjectiveVector > & _f)
    {
      for (unsigned int i=0 ; i<_f.size(); i++)
        {
          bool dom = false;
          for (unsigned int j=0; j<_f.size(); j++)
            if (i != j && paretoComparator(_f[i],_f[j]))
              {
                dom = true;
                break;
              }
          if (dom)
            {
              _f[i] = _f.back();
              _f.pop_back();
              i--;
            }
        }
    }


    /**
     * Prenormalization
     * @param _f a Pareto set
     */
    void prenormalize (const std::vector< ObjectiveVector > & _f)
    {
      vect_min_val.clear();
      vect_max_val.clear();

      for (unsigned int i=0 ; i<ObjectiveVector::nObjectives(); i++)
        {
          float min_val = _f.front()[i], max_val = min_val;
          for (unsigned int j=1 ; j<_f.size(); j++)
            {
              if (_f[j][i] < min_val)
                min_val = _f[j][i];
              if (_f[j][i]>max_val)
                max_val = _f[j][i];
            }
          vect_min_val.push_back(min_val);
          vect_max_val.push_back (max_val);
        }
    }


    /**
     * Normalization
     * @param _f a Pareto set
     */
    void normalize (std::vector< ObjectiveVector > & _f)
    {
      for (unsigned int i=0 ; i<ObjectiveVector::nObjectives(); i++)
        for (unsigned int j=0; j<_f.size(); j++)
          _f[j][i] = (_f[j][i] - vect_min_val[i]) / (vect_max_val[i] - vect_min_val[i]);
    }


    /**
     * Computation of the union of _f1 and _f2 in _f
     * @param _f1 the first Pareto set
     * @param _f2 the second Pareto set
     * @param _f the final Pareto set
     */
    void computeUnion(const std::vector< ObjectiveVector > & _f1, const std::vector< ObjectiveVector > & _f2, std::vector< ObjectiveVector > & _f)
    {
      _f = _f1 ;
      for (unsigned int i=0; i<_f2.size(); i++)
        {
          bool b = false;
          for (unsigned int j=0; j<_f1.size(); j ++)
            if (_f1[j] == _f2[i])
              {
                b = true;
                break;
              }
          if (! b)
            _f.push_back(_f2[i]);
        }
    }


    /**
     * How many in niche
     */
    unsigned int howManyInNicheOf (const std::vector< ObjectiveVector > & _f, const ObjectiveVector & _s, unsigned int _size)
    {
      unsigned int n=0;
      for (unsigned int i=0 ; i<_f.size(); i++)
        {
          if (euclidianDistance(_f[i], _s) < (_s.size() / (double) _size))
            n++;
        }
      return n;
    }


    /**
     * Euclidian distance
     */
    double euclidianDistance (const ObjectiveVector & _set1, const ObjectiveVector & _to, unsigned int _deg = 2)
    {
      double dist=0;
      for (unsigned int i=0; i<_set1.size(); i++)
        dist += pow(fabs(_set1[i] - _to[i]), (int)_deg);
      return pow(dist, 1.0 / _deg);
    }

  };

#endif /*MOEOENTROPYMETRIC_H_*/
