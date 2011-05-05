// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoStochasticUniversalSelect.h
// (c) Maarten Keijzer 2003
/*
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
             mkeijzer@cs.vu.nl
 */
//-----------------------------------------------------------------------------

#ifndef eoStochasticUniversalSelect_h
#define eoStochasticUniversalSelect_h

//-----------------------------------------------------------------------------

#include <utils/eoRNG.h>
#include <eoSelectOne.h>
#include <utils/selectors.h>
#include <eoPop.h>

/** eoStochasticUniversalSelect: select an individual proportional to her stored fitness
    value, but in contrast with eoStochasticUniversalSelect, get rid of most finite sampling effects
    by doing all selections in one go, using a single random number.

    @ingroup Selectors
*/
template <class EOT> class eoStochasticUniversalSelect: public eoSelectOne<EOT>
{
public:
  /// Sanity check
  eoStochasticUniversalSelect(const eoPop<EOT>& pop = eoPop<EOT>())
  {
    if (minimizing_fitness<EOT>())
      throw std::logic_error("eoStochasticUniversalSelect: minimizing fitness");
  }

  void setup(const eoPop<EOT>& _pop)
  {
      if (_pop.size() == 0) return;

      std::vector<typename EOT::Fitness> cumulative(_pop.size());

      cumulative[0] = _pop[0].fitness();
      for (unsigned i = 1; i < _pop.size(); ++i)
      {
          cumulative[i] = _pop[i].fitness() + cumulative[i-1];
      }

      indices.reserve(_pop.size());
      indices.resize(0);

      double fortune = rng.uniform() * cumulative.back();
      double step = cumulative.back() / double(_pop.size());

      unsigned i = std::upper_bound(cumulative.begin(), cumulative.end(), fortune) - cumulative.begin();

      while (indices.size() < _pop.size()) {

          while (cumulative[i] < fortune) {i++;} // linear search is good enough as we average one step each time

          indices.push_back(i);
          fortune += step;
          if (fortune >= cumulative.back()) { // start at the beginning
              fortune -= cumulative.back();
              i = 0;
          }
      }
      // shuffle
      for (int i = indices.size() - 1; i > 0; --i) {
          int j = rng.random(i+1);
          std::swap(indices[i], indices[j]);
      }
  }

  /** do the selection,
   */
  const EOT& operator()(const eoPop<EOT>& _pop)
  {
      if (indices.empty()) setup(_pop);

      unsigned index = indices.back();
      indices.pop_back();
      return _pop[index];
  }

private :

  typedef std::vector<unsigned> IndexVec;
  IndexVec indices;
};
/** @example t-eoRoulette.cpp
 */

#endif
