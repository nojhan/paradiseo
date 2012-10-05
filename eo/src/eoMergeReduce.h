/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoMergeReduce.h
   (c) Maarten Keijzer, GeNeura Team, 2000

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
 */
//-----------------------------------------------------------------------------

#ifndef _eoMergeReduce_h
#define _eoMergeReduce_h


//-----------------------------------------------------------------------------
#include <eoPop.h>
#include <eoFunctor.h>
#include <eoMerge.h>
#include <eoReduce.h>
#include <eoReplacement.h>
#include <utils/eoHowMany.h>
//-----------------------------------------------------------------------------
/**
Replacement strategies that combine en eoMerge and an eoReduce.

@class eoMergeReduce, the base (pure abstract) class
@class eoPlusReplacement the ES plus strategy
@class eoCommaReplacement the ES comma strategy
*/

/**
eoMergeReduce: abstract replacement strategy that is just an application of
an embedded merge, followed by an embedded reduce
@ingroup Replacors
*/
template <class EOT>
class eoMergeReduce : public eoReplacement<EOT>
{
    public:
        eoMergeReduce(eoMerge<EOT>& _merge, eoReduce<EOT>& _reduce) :
        merge(_merge), reduce(_reduce)
        {}

        virtual void operator()(eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
        {
            merge(_parents, _offspring); // parents untouched, result in offspring
            reduce(_offspring, _parents.size());
            _parents.swap(_offspring);
        }

    private :
        eoMerge<EOT>& merge;
        eoReduce<EOT>& reduce;
};

/**
ES type of replacement strategy: first add parents to population, then truncate
@ingroup Replacors
*/
template <class EOT>
class eoPlusReplacement : public eoMergeReduce<EOT>
{
    public :
        eoPlusReplacement() : eoMergeReduce<EOT>(plus, truncate) {}

    private :
        eoPlus<EOT> plus;
        eoTruncate<EOT> truncate;
};

/**
ES type of replacement strategy: ignore parents, truncate offspring
@ingroup Replacors
*/
template <class EOT>
class eoCommaReplacement : public eoMergeReduce<EOT>
{
    public :
        eoCommaReplacement() : eoMergeReduce<EOT>(no_elite, truncate) {}

        virtual void operator()(eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
        {
            // There must be more offsprings than parents, or else an exception will be raised
            assert( _offspring.size() >= _parents.size() );

            eoMergeReduce<EOT>::operator()( _parents, _offspring );
        }

    private :
        eoNoElitism<EOT> no_elite;
        eoTruncate<EOT> truncate;
};

/**
EP type of replacement strategy: first add parents to population,
   then truncate using EP tournament
@ingroup Replacors
*/
template <class EOT>
class eoEPReplacement : public eoMergeReduce<EOT>
{
public :
  eoEPReplacement(int _tSize) : eoMergeReduce<EOT>(plus, truncate), truncate(_tSize)
    //  {truncate.setSize(_tSize);}
  {}
private :
  eoPlus<EOT> plus;
  eoEPReduce<EOT> truncate;
};



#endif
