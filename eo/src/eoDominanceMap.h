/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoDominanceMap.h
   (c) Maarten Keijzer, Marc Schoenauer, 2001

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoDominanceMap_h
#define eoDominanceMap_h

#include <eoFunctor.h>
#include <eoPop.h>

/**
  eoDominanceMap, utility class to calculate and maintain a map (vector<vector<bool> >) of pareto dominance statistics.

  It is set up such that

  if map[i][j] == true
  then i dominates j

  The dominance map can be used to perform pareto ranking (eoParetoRanking) or non dominated sorting.
  For the latter, the remove() member function might come in handy.

  \todo make it an eoStat?
*/
template <class EoType>
class eoDominanceMap : public eoUF<const eoPop<EoType>&, void>, public std::vector<std::vector<bool> >
{
  public :
  /**
    Ctor that needs to know what objectives are supposed to be maximized and which are to be minimized.
    optional argument, a tolerance. This designates within which tolerance objective values are considered to be equal
  */
  eoDominanceMap(const std::vector<bool>& _maximizes, double _tol = 1e-6) : maximizes(_maximizes), tol(_tol) {}

  /**
    Clears the map
  */
  void clear()
  {
    std::vector<std::vector<bool> >::clear();
    fitnesses.clear();
  }

  /**
    Update or create the dominance map
  */
  void operator()(const eoPop<EoType>& _pop)
  {
    if (fitness.size() != _pop.size())
    { // fresh start
      setup(_pop);
      return;
    }
    // else just update the guys that have changed

    update(_pop);
  }

  /// update the map with the population
  void update(const eoPop<EoType>& _pop)
  {
    for (unsigned i = 0; i < _pop.size(); ++i)
    {
      if (fitness[i] == _pop[i].fitness())
      {
        continue;
      }
      // it's a new guy, update rows and columns
      fitness[i] = _pop[i].fitness();

      for (unsigned j = 0; j < _pop.size(); ++j)
      {
        if (i == j)
          continue;

        switch (dominates(_pop[i].fitness(), _pop[j].fitness()))
        {
          case a_dominates_b :
          {
            operator[](i)[j] = true;
            operator[](j)[i] = false;
            break;
          }
          case b_dominates_a :
          {
            operator[](i)[j] = false;
            operator[](j)[i] = true;
            break;
          }
          default :
          {
            operator[](i)[j] = false;
            operator[](j)[i] = false;
          }
        }
      }
    }
  }

  /**
    Removes the domination info for a given individual, thus nothing dominates it and it dominates nothing.
  */
  void remove(unsigned i)
  {
    for (unsigned j = 0; j < size(); ++j)
    {
      operator[](i)[j] = false; // clear row
      operator[](j)[i] = false; // clear col
    }
  }

  /**
    Create domination matrix from scratch. Complexity O(N^2)
  */
  void setup(const eoPop<EoType>& _pop)
  {
    fitness.resize(_pop.size());
    resize(_pop.size());

    for (unsigned i = 0; i < _pop.size(); ++i)
    {
      fitness[i] = _pop[i].fitness();
      operator[](i).resize(_pop.size(), false);

      for (unsigned j = 0; j < i; ++j)
      {
        switch(dominates(fitness[i], fitness[j]))
        {
          case a_dominates_b :
          {
            operator[](i)[j] = true;
            operator[](j)[i] = false;
            break;
          }
          case b_dominates_a :
          {
            operator[](i)[j] = false;
            operator[](j)[i] = true;
            break;
          }
          default :
          {
            operator[](i)[j] = false;
            operator[](j)[i] = false;
          }
        }
      }
    }
  }

  /**
    For all elements, returns the no. of elements that dominate the element
    Thus: lower is better (and 0 is the front).
    It returns a vector<double> cuz that
    makes subsequent manipulation that much easier
  */
  vector<double> sum_dominators() const
  {
    vector<double> result(size(), 0.0);

    for (unsigned i = 0; i < size(); ++i)
    {
      for (unsigned j = 0; j < size(); ++j)
      {
        if (operator[](i)[j]) // i dominates j
          result[j]++;
      }
    }

    return result;
  }

  /**
    For all elements, returns the number of elements that the element dominates
    Thus: higher is better
    It returns a vector<double> cuz that
    makes subsequent manipulation that much easier
  */
  vector<double> sum_dominants() const
  {
    vector<double> result(size(), 0.0);

    for (unsigned i = 0; i < size(); ++i)
    {
      for (unsigned j = 0; j < size(); ++j)
      {
        if (operator[](i)[j]) // i dominates j
          result[i]++;
      }
    }

    return result;
  }

  /**
    make a distinction between no domination (a better than b on one objective and worse on another)
    and equality: a == b
  */
  enum dom_res {a_dominates_b, b_dominates_a, no_domination, a_equals_b};

  /**
    a dominates b if it is better in one objective and not worse in any of the others
  */
  dom_res dominates(typename EoType::Fitness a, typename EoType::Fitness b)
  {
    dom_res result = a_equals_b;

    for (unsigned i = 0; i < a.size(); ++i)
    {
      double aval = a[i];
      double bval = b[i];

      if (!maximizes[i])
      {
        aval = -aval;
        bval = -bval;
      }

      // check if unequal, in the case they're 'equal' just go to the next
      if (fabs(aval - bval) > tol)
      {
        if (aval > bval)
        {
          if (result == b_dominates_a)
          {
            return no_domination; // when on another objective b dominated a, they do not dominate each other
          }
          // else continue comparing

          result = a_dominates_b;
        }
        else // bval > aval
        {
          if (result == a_dominates_b)
          {
            return no_domination; // done
          }

          result = b_dominates_a;
        }
      }
    }

    return result;
  }

  private :


  vector<bool> maximizes;
  double tol;
  vector<typename EoType::Fitness> fitness;
};

#endif