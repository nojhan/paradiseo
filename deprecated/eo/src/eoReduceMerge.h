/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoReduceMerge.h
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

#ifndef _eoReduceMerge_h
#define _eoReduceMerge_h


//-----------------------------------------------------------------------------
#include <eoPop.h>
#include <eoFunctor.h>
#include <eoMerge.h>
#include <eoReduce.h>
#include <eoReplacement.h>
#include <utils/eoHowMany.h>
//-----------------------------------------------------------------------------

/** @addtogroup Replacors
 * @{
 */

/**
eoReduceMerge: Replacement strategies that start by reducing the parents,
               then merge with the offspring

This is the way to do SSGA: the offspring gets inserted in the population
even if it is worse than anybody else.

@see eoReduceMerge
@see eoSSGAWorseReplacement
@see eoSSGADetTournamentReplacement
@see eoSSGAStochTournamentReplacement
*/
template <class EOT>
class eoReduceMerge : public eoReplacement<EOT>
{
    public:
        eoReduceMerge(eoReduce<EOT>& _reduce, eoMerge<EOT>& _merge) :
        reduce(_reduce), merge(_merge)
        {}

        void operator()(eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
        {
          if (_parents.size() < _offspring.size())
            throw std::logic_error("eoReduceMerge: More offspring than parents!\n");
          reduce(_parents, _parents.size() - _offspring.size());
          merge(_offspring, _parents);
        }

    private :
        eoReduce<EOT>& reduce;
        eoMerge<EOT>& merge;
};

/**
SSGA replace worst. Is an eoReduceMerge.
*/
template <class EOT>
class eoSSGAWorseReplacement : public eoReduceMerge<EOT>
{
    public :
        eoSSGAWorseReplacement() : eoReduceMerge<EOT>(truncate, plus) {}

    private :
        eoLinearTruncate<EOT> truncate;
        eoPlus<EOT> plus;
};

/**
SSGA deterministic tournament replacement. Is an eoReduceMerge.
*/
template <class EOT>
class eoSSGADetTournamentReplacement : public eoReduceMerge<EOT>
{
    public :
        eoSSGADetTournamentReplacement(unsigned _t_size) :
          eoReduceMerge<EOT>(truncate, plus), truncate(_t_size) {}

    private :
        eoDetTournamentTruncate<EOT> truncate;
        eoPlus<EOT> plus;
};

/** SSGA stochastic tournament replacement. Is an eoReduceMerge.
It much cleaner to insert directly the offspring in the parent population,
but it is NOT equivalent in case of more than 1 offspring as already
replaced could be removed , which is not possible in the eoReduceMerge
So what the heck ! */
template <class EOT>
class eoSSGAStochTournamentReplacement : public eoReduceMerge<EOT>
{
    public :
        eoSSGAStochTournamentReplacement(double _t_rate) :
          eoReduceMerge<EOT>(truncate, plus), truncate(_t_rate) {}

    private :
        eoStochTournamentTruncate<EOT> truncate;
        eoPlus<EOT> plus;
};

/** @} */
#endif
