// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRankingSelect.h
// (c) GeNeura Team, 1998, Maarten Keijzer 2000, Marc Schoenauer 2001
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

    Contact: Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoRankingSelect_h
#define eoRankingSelect_h

//-----------------------------------------------------------------------------

#include <eoSelectFromWorth.h>
#include <eoRanking.h>

/** eoRankingSelect: select an individual by roulette wheel on its rank
 *  is an eoRouletteWorthSelect, i.e. a selector using a std::vector of worthes
 *  rather than the raw fitness (see eoSelectFromWorth.h)
 *  uses an internal eoRanking object which is an eoPerf2Worth<EOT, double>
 *
 * @ingroup Selectors
*/
template <class EOT>
class eoRankingSelect: public eoRouletteWorthSelect<EOT, double>
{
public:
  /** Ctor:
   *  @param _p the selective pressure, should be in [1,2] (2 is the default)
   *  @param _e exponent (1 == linear)
   */
  eoRankingSelect(double _p = 2.0, double _e=1.0):
    eoRouletteWorthSelect<EOT, double>(ranking), ranking(_p, _e) {}

private :
  eoRanking<EOT> ranking;          // derived from eoPerf2Worth
};

#endif
