/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoNDSorting.h
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

#ifndef eoNDSorting_h
#define eoNDSorting_h

#include <algorithm>
#include <eoPerf2Worth.h>
#include <eoDominanceMap.h>

/**
  Non dominated sorting
*/
template <class EOT>
class eoNDSorting : public eoPerf2Worth<EOT, double>
{
  public :

    eoNDSorting(eoDominanceMap<EOT>& _dominanceMap, double _nicheSize) :
      eoPerf2Worth<EOT, double>(), dominanceMap(_dominanceMap), nicheSize(_nicheSize) {}

    void operator()(const eoPop<EOT>& _pop)
    {
      dominanceMap(_pop);

      vector<bool> excluded(_pop.size(), false);

      value().resize(_pop.size());

      bool finished = false;

      int dominance_level = 0;

      while(!finished)
      {
        vector<double> ranks = dominanceMap.sum_dominators();

        vector<unsigned> current_front;
        current_front.reserve(_pop.size());

        finished = true;

        for (unsigned i = 0; i < _pop.size(); ++i)
        {
          if (excluded[i])
          {
            continue; // next please
          }

          if (ranks[i] < 1e-6)
          {// it's part of the current front
            excluded[i] = true;
            current_front.push_back(i);
            dominanceMap.remove(i); // remove from consideration
          }
          else
          {
            finished = false; // we need another go
          }
        }

        // Now we have the indices to the current front in current_front, do the niching

        // As I don't have my reference text with me some homespun savagery

        ranks = dominanceMap.sum_dominants(); // how many do you dominate

        double max_rank = *std::max_element(ranks.begin(), ranks.end());

        for (unsigned i = 0; i < current_front.size(); ++i)
        {
          // punish the ones that dominate the most individuals (sounds strange huh?)
          value()[current_front[i]] = dominance_level + ranks[i] / (max_rank + 1);
        }


        dominance_level++; // go to the next front
      }

      // now all that's left to do is to transform lower rank into higher worth
      double max_fitness = *std::max_element(value().begin(), value().end());

      for (unsigned i = 0; i < value().size(); ++i)
      {
        value()[i] = max_fitness - value()[i];
      }

    }

  private :

  eoDominanceMap<EOT>& dominanceMap;
  double nicheSize;
};

#endif