/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoReduceMergeReduce.h
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

#ifndef _eoReduceMergeReduce_h
#define _eoReduceMergeReduce_h


//-----------------------------------------------------------------------------
#include <eoPop.h>
#include <eoFunctor.h>
#include <eoMerge.h>
#include <eoReduce.h>
#include <utils/eoHowMany.h>
//-----------------------------------------------------------------------------

/**
eoReduceMergeReduce is an eoReplacement:
- saves possible elite parents
- reduces rest of parents
- reduces offspring
- merges reduced populations
- reduces resulting merged pop if necessary

   @ingroup Replacors
*/
template <class EOT>
class eoReduceMergeReduce : public eoReplacement<EOT>
{
public:
  eoReduceMergeReduce(eoHowMany _howManyElite,
                      bool _strongElitism,
                      eoHowMany _howManyReducedParents,
                      eoReduce<EOT> & _reduceParents,
                      eoHowMany _howManyReducedOffspring,
                      eoReduce<EOT> & _reduceOffspring,
                      eoReduce<EOT> & _reduceFinal) :
    howManyElite(_howManyElite),
    strongElitism(_strongElitism),
    howManyReducedParents(_howManyReducedParents),
    howManyReducedOffspring (_howManyReducedOffspring),
    reduceParents(_reduceParents),
    reduceOffspring(_reduceOffspring),
    reduceFinal(_reduceFinal)
  {}

    void operator()(eoPop<EOT> & _parents, eoPop<EOT> & _offspring)
    {
      eoPop<EOT> temp;
      unsigned int finalPopSize = _parents.size();
      unsigned int offSize = _offspring.size();

      unsigned int elite = howManyElite(finalPopSize);
      if (elite)                   // some parents MUST be saved somewhere
        {
          temp.resize(elite);
          _parents.nth_element(elite);
          std::copy(_parents.begin(), _parents.begin()+elite, temp.begin());
          _parents.erase(_parents.begin(), _parents.begin()+elite);
        }

      // the reduce steps. First the parents
      unsigned reducedParentSize = howManyReducedParents(_parents.size());
      if (!reducedParentSize)
        _parents.clear();
      else if (reducedParentSize != _parents.size())
        reduceParents(_parents, reducedParentSize);

      // then the offspring
      unsigned reducedOffspringSize = howManyReducedOffspring(offSize);
      if (!reducedOffspringSize)
        throw std::runtime_error("No offspring left after reduction!");
      if (reducedOffspringSize != offSize) // need reduction
        reduceOffspring(_offspring, reducedOffspringSize);

      // now merge reduced populations
      _parents.resize(reducedParentSize + _offspring.size());
      std::copy(_offspring.begin(), _offspring.end(),
                _parents.begin()+reducedParentSize);

      // reduce the resulting population
      // size depstd::ends on elitism
      if (elite && strongElitism)
        {
          if (_parents.size() != finalPopSize-elite)
            reduceFinal(_parents, finalPopSize-elite);
          // and put back the elite
          unsigned oldPSize = _parents.size();
          _parents.resize(_parents.size()+elite);
          std::copy(temp.begin(), temp.end(), _parents.begin()+oldPSize);
        }
      else
        {                   // only reduce final pop to right size
          if (_parents.size() != finalPopSize)
            reduceFinal(_parents, finalPopSize);
          if (elite)       // then treat weak elitism
            {
              unsigned toSave = 0;
              _parents.sort();
              EOT & eoLimit = _parents[elite-1];
              unsigned index=0;
              while ( (temp[index++] > eoLimit) && (index < temp.size()) )
                toSave++;
              if (toSave)
                for (unsigned i=0; i<toSave; i++)
                  _parents[finalPopSize-1-i] = temp[i];
            }
        }
    }

private:
  eoHowMany howManyElite;          // if 0, no elitism at all
  bool strongElitism;              // if false -> weak estd::listism
  eoHowMany howManyReducedParents; // if 0, no parent in final replacement
  eoHowMany howManyReducedOffspring; // if 0, std::runtime_error
  // the reducers
  eoReduce<EOT> & reduceParents;
  eoReduce<EOT> & reduceOffspring;
  eoReduce<EOT> & reduceFinal;
};

#endif
