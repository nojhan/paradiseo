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

#include <utils/eoRNG.h>
#include <utils/selectors.h>
#include <eoSelectOne.h>
#include <eoPop.h>

/** eoProportionalSelect: select an individual proportional to her stored fitness
    value

    Changed the algorithm to make use of a cumulative array of fitness scores,
    This changes the algorithm from O(n) per call to  O(log n) per call. (MK)

    @ingroup Selectors
*/
template <class EOT> class eoProportionalSelect: public eoSelectOne<EOT>
{
public:
  /// Sanity check
  eoProportionalSelect(const eoPop<EOT>& /*pop*/ = eoPop<EOT>())
  {
    if (minimizing_fitness<EOT>())
      throw std::logic_error("eoProportionalSelect: minimizing fitness");
  }

  void setup(const eoPop<EOT>& _pop)
  {
      if (_pop.size() == 0) return;

      cumulative.resize(_pop.size());
      cumulative[0] = _pop[0].fitness();

      for (unsigned i = 1; i < _pop.size(); ++i)
      {
          cumulative[i] = _pop[i].fitness() + cumulative[i-1];
      }
  }

  /** do the selection,
   */
  const EOT& operator()(const eoPop<EOT>& _pop)
  {
      if (cumulative.size() == 0) setup(_pop);

      double fortune = rng.uniform() * cumulative.back();
      typename FitVec::iterator result = std::upper_bound(cumulative.begin(), cumulative.end(), fortune);
      return _pop[result - cumulative.begin()];
  }

private :

  typedef std::vector<typename EOT::Fitness> FitVec;
  FitVec cumulative;
};
/** @example t-eoRoulette.cpp
 */

#endif
