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
  eoDominanceMap, utility class to calculate and maintain a map (std::vector<std::vector<bool> >) of pareto dominance statistics.

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
public:

    /** Clears the map */
    void clear() {
        std::vector<std::vector<bool> >::clear();
        fitness.clear();
    }

  /**
    Update or create the dominance map
  */
  void operator()(const eoPop<EoType>& _pop)
  {
    setup(_pop);
    return;
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
        if (_pop[i].fitness().dominates(_pop[j].fitness()))
        {
          operator[](i)[j] = true;
          operator[](j)[i] = false;
        }
        else if (_pop[j].fitness().dominates(_pop[i].fitness()))
        {
          operator[](i)[j] = false;
          operator[](j)[i] = true;
        }
        else
        {
          operator[](i)[j] = false;
          operator[](j)[i] = false;
        }
      }
    }
  }

  /**
    For all elements, returns the no. of elements that dominate the element
    Thus: lower is better (and 0 is the front).
    It returns a std::vector<double> cuz that
    makes subsequent manipulation that much easier
  */
  std::vector<double> sum_dominators() const
  {
    std::vector<double> result(size(), 0.0);

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
    It returns a std::vector<double> cuz that
    makes subsequent manipulation that much easier
  */
  std::vector<double> sum_dominants() const
  {
    std::vector<double> result(size(), 0.0);

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

  private :


  std::vector<typename EoType::Fitness> fitness;
};

#endif
