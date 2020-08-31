// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoProportionalSelect.h
// (c) GeNeura Team, 1998 - EEAAX 1999, Maarten Keijzer 2000
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
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoProportionalSelect_h
#define eoProportionalSelect_h

//-----------------------------------------------------------------------------

#include <iterator>
#include <utility>

#include "utils/eoRNG.h"
#include "utils/selectors.h"
#include "eoSelectOne.h"
#include "eoPop.h"

/** eoProportionalSelect: select an individual proportional to her stored fitness
    value

    Changed the algorithm to make use of a cumulative array of fitness scores,
    This changes the algorithm from O(n) per call to  O(log n) per call. (MK)

    @ingroup Selectors
*/
template <class EOT>
class eoProportionalSelect: public eoSelectOne<EOT>
{
public:
  /// Sanity check
  eoProportionalSelect(const eoPop<EOT>& /*pop*/ = eoPop<EOT>())
  {
      if (minimizing_fitness<EOT>()) {
          std::string msg = "eoProportionalSelect cannot be used with minimizing fitness";
          eo::log << eo::errors << "ERROR: " << msg << std::endl;
          throw eoException(msg);
      }
  }

  void setup(const eoPop<EOT>& _pop)
  {
      if (_pop.size() == 0) {
          eo::log << eo::warnings << "Warning: eoProportionalSelect setup called on an empty population." << std::endl;
          return;
      }
      assert(not _pop[0].invalid());

      const typename EOT::Fitness min_fit
          = std::min_element( std::begin(_pop), std::end(_pop) )
              ->fitness();

      cumulative.clear();
      cumulative.push_back(_pop[0].fitness() - min_fit);

      for (unsigned i = 1; i < _pop.size(); ++i) {
          assert(not _pop[i].invalid());
          cumulative.push_back(cumulative.back() + _pop[i].fitness() - min_fit);
      }
      assert(cumulative.size() == _pop.size());
  }

  /** do the selection,
   */
  const EOT& operator()(const eoPop<EOT>& _pop)
  {
      if (cumulative.size() == 0) setup(_pop);

      double frac = rng.uniform();
      double fortune = frac * cumulative.back();
      typename FitVec::iterator result
          = std::upper_bound(cumulative.begin(), cumulative.end(), fortune);

      assert(fortune <= cumulative.back());

      if(result - cumulative.begin() == _pop.size()) {
            return _pop.back();
      } else {
          return _pop[result - cumulative.begin()];
      }
  }

private :

  typedef std::vector<typename EOT::Fitness> FitVec;
  FitVec cumulative;
public:
  virtual std::string className() const {return "eoProportionalSelect";}
};
/** @example t-eoRoulette.cpp
 */

#endif
