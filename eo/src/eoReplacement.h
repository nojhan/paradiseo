/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoReplacement.h 
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

#ifndef _eoReplacement_h
#define _eoReplacement_h


//-----------------------------------------------------------------------------
#include <eoPop.h>
#include <eoFunctor.h>
#include <eoMerge.h>
#include <eoReduce.h>
//-----------------------------------------------------------------------------

/** 
eoReplacement: High level strategy for creating a new generation 
from parents and offspring. This is a combination of eoMerge and eoReduce,
so there is an implementation called eoMergeReduce that can be found below

  @see eoMerge, eoReduce, eoMergeReduce
*/
template<class EOT>
class eoReplacement : public eoBinaryFunctor<void, const eoPop<EOT>&, eoPop<EOT>&> 
{};

/**
no replacement
*/
template <class EOT>
class eoNoReplacement : public eoReplacement<EOT>
{
    public :
        /// do nothing
        void operator()(const eoPop<EOT>&, eoPop<EOT>&)
        {}
};

/**
eoMergeReduce: special replacement strategy that is just an application of an embedded merge, 
followed by an embedded reduce
*/
template <class EOT>
class eoMergeReduce : public eoReplacement<EOT>
{
    public:
        eoMergeReduce(eoMerge<EOT>& _merge, eoReduce<EOT>& _reduce) :
        merge(_merge), reduce(_reduce)
        {}

        void operator()(const eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
        {
            merge(_parents, _offspring);
            reduce(_offspring, _parents.size());
        }

    private :
        eoMerge<EOT>& merge;
        eoReduce<EOT>& reduce;
};

/**
ES type of replacement strategy: first add parents to population, then truncate
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
*/
template <class EOT>
class eoCommaReplacement : public eoMergeReduce<EOT>
{
    public :
        eoCommaReplacement() : eoMergeReduce<EOT>(no_elite, truncate) {}

    private :
        eoNoElitism<EOT> no_elite;
        eoTruncate<EOT> truncate;
};


#endif
