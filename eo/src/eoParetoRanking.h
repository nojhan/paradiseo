/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoParetoRanking.h
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

#ifndef eoParetoRanking_h
#define eoParetoRanking_h

#include <eoPerf2Worth.h>
#include <eoDominanceMap.h>

/**
  Straightforward pareto ranking. Every individual gets a rank according to the number
  of elements it dominates. Note that without niching, this technique will usually not
  find the whole front of non-dominated solutions, but will quite likely converge
  on a single spot on the front.
*/
template <class EOT>
class eoParetoRanking : public eoPerf2WorthCached<EOT, double>
{
public:

    using eoParetoRanking< EOT>::value;

    eoParetoRanking(eoDominanceMap<EOT>& _dominanceMap)
        : eoPerf2WorthCached<EOT, double>(), dominanceMap(_dominanceMap)
        {}

    void calculate_worths(const eoPop<EOT>& _pop)
    {
      dominanceMap(_pop);
      value() = dominanceMap.sum_dominators(); // get rank: 0 means part of current front

      // calculate maximum
      double maxim = *std::max_element(value().begin(), value().end());

      // higher is better, so invert the value
      for (unsigned i = 0; i < value().size(); ++i)
      {
        value()[i] = maxim - value()[i];
      }

    }

  private :

  eoDominanceMap<EOT>& dominanceMap;
};

#endif
