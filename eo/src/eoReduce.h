// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoReduce.h
//   Base class for population-merging classes
// (c) GeNeura Team, 1998
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
 */
//-----------------------------------------------------------------------------

#ifndef eoReduce_h
#define eoReduce_h

//-----------------------------------------------------------------------------

#include <iostream>

// EO includes
#include <eoPop.h>     // eoPop
#include <eoFunctor.h>  // eoReduce

/**
 * eoReduce: .reduce the new generation to the specified size
*/

template<class Chrom> class eoReduce: public eoBinaryFunctor<void, eoPop<Chrom>&, unsigned>
{};

template <class EOT> class eoTruncate : public eoReduce<EOT>
{
    void operator()(eoPop<EOT>& _newgen, unsigned _newsize)
    {
        if (_newgen.size() == _newsize)
            return;

        _newgen.nth_element(_newsize);
        _newgen.resize(_newsize);
    }
};

//-----------------------------------------------------------------------------

#endif //eoInsertion_h
