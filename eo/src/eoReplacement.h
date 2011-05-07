/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoReplacement.h
   (c) Maarten Keijzer, Marc Schoenauer, GeNeura Team, 2000

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

#ifndef _eoReplacement_h
#define _eoReplacement_h


//-----------------------------------------------------------------------------
#include <eoPop.h>
#include <eoFunctor.h>
#include <eoMerge.h>
#include <eoReduce.h>
#include <utils/eoHowMany.h>
//-----------------------------------------------------------------------------

/**
---
The eoMergeReduce, combination of eoMerge and eoReduce, can be found
in file eoMergeReduce.h

The eoReduceMergeReduce that reduces the parents and the offspring,
merges the 2 reduced populations, and eventually reduces that final
population, can be found in eoReduceMergeReduce.h

LOG
---
Removed the const before first argument: though it makes too many classes
with the same interface, it allows to minimize the number of actual copies
by choosing the right destination
I also removed the enforced "swap" in the eoEasyAlgo and hence the generational
replacement gets a class of its own that only does the swap (instead of the
eoNoReplacement that did nothing, relying on the algo to swap popualtions).
MS 12/12/2000

  @see eoMerge, eoReduce, eoMergeReduce, eoReduceMerge

@class eoReplacement,                    base (pure abstract) class
@class eoGenerationalReplacement,        as it says ...
@class eoWeakElitistReplacement          a wrapper to add elitism

*/

/** The base class for all replacement functors.

NOTE: two eoPop as arguments
the resulting population should be in the first argument (replace
parents by offspring)! The second argument can contain any rubbish

 @ingroup Replacors
 */
template<class EOT>
class eoReplacement : public eoBF<eoPop<EOT>&, eoPop<EOT>&, void>
{};

/**
generational replacement == swap populations

 @ingroup Replacors
*/
template <class EOT>
class eoGenerationalReplacement : public eoReplacement<EOT>
{
    public :
  /// swap
  void operator()(eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
  {
    _parents.swap(_offspring);
  }
};

/**
eoWeakElitistReplacement: a wrapper for other replacement procedures.
Copies in the new pop the best individual from the old pop,
AFTER normal replacement, if the best of the new pop is worse than the best
of the old pop. Removes the worse individual from the new pop.
This could be changed by adding a selector there...

 @ingroup Replacors
*/
template <class EOT>
class eoWeakElitistReplacement : public eoReplacement<EOT>
{
public :
  typedef typename EOT::Fitness Fitness;

  // Ctor, takes an eoReplacement
  eoWeakElitistReplacement(eoReplacement<EOT> & _replace) :
    replace(_replace) {}

  /// do replacement
  void operator()(eoPop<EOT>& _pop, eoPop<EOT>& _offspring)
  {
    const EOT oldChamp = _pop.best_element();
    replace(_pop, _offspring);     // "normal" replacement, parents are the new
    if (_pop.best_element() < oldChamp) // need to do something
      {
        typename eoPop<EOT>::iterator itPoorGuy = _pop.it_worse_element();
        (*itPoorGuy) = oldChamp;
      }
  }
private:
  eoReplacement<EOT> & replace;
};

#endif
