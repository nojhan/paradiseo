/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoMGGReplacement.h
   (c) Maarten Keijzer, Marc Schoenauer, 2002

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

    Contact: Marc.Schoenauer@inria.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoMGGReplacement_h
#define _eoMGGReplacement_h


//-----------------------------------------------------------------------------
#include <eoPop.h>
#include <eoFunctor.h>
#include <eoMerge.h>
#include <eoReduce.h>
#include <utils/eoHowMany.h>
#include <eoReduceSplit.h>
//-----------------------------------------------------------------------------

/**
eoMGGReplacement is an eoReplacement:
- choose N (2) parents RANDOMLY - remove them from the parent population
- select best offspring, add to parents
- merge (other?) offspring and the N removed parents
- select best N-1 of this merged population (detTournament only at the moment)
- put them back into parent population

@ingroup Replacors
*/

template <class EOT>
class eoMGGReplacement : public eoReplacement<EOT>
{
public:
  eoMGGReplacement(eoHowMany _howManyEliminatedParents = eoHowMany(2, false),
                   unsigned _tSize=2) :
    // split truncates the parents and returns eliminated parents
    split(_howManyEliminatedParents, true),
    tSize(_tSize)
  {
    if (tSize < 2)
      {
          eo::log << eo::warnings << "Warning, Size for eoDetTournamentTruncateSplit adjusted to 2" << std::endl;
        tSize = 2;
      }
  }

    void operator()(eoPop<EOT> & _parents, eoPop<EOT> & _offspring)
    {
      eoPop<EOT> temp;
      split(_parents, temp);
      unsigned toKeep = temp.size(); // how many to keep from merged populations
      // minimal check
      if (toKeep < 2)
        throw std::runtime_error("Not enough parents killed in eoMGGReplacement");

      // select best offspring
      typename eoPop<EOT>::iterator it = _offspring.it_best_element();
      // add to parents
      _parents.push_back(*it);
      // remove from offspring
      _offspring.erase(it);

      // merge temp into offspring
      plus(temp, _offspring);

      // repeatedly add selected offspring to parents
      for (unsigned i=0; i<toKeep-1; i++)
        {
          // select
          it = deterministic_tournament(_offspring.begin(), _offspring.end(), tSize);
          // add to parents
          _parents.push_back(*it);
          // remove from offspring
          _offspring.erase(it);
        }
    }

private:
  eoLinearTruncateSplit<EOT> split; // few parents to truncate -> linear
  eoPlus<EOT> plus;
  unsigned int tSize;
};

#endif
