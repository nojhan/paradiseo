/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoInserter.h
    Abstract population insertion operator, which is used by the eoGeneralOps
    to insert the results in the (intermediate) population. This file also
    contains the definitions of a derived classes that implements a back inserter,
    probably the only efficient inserter for populations of type vector.

 (c) Maarten Keijzer (mkeijzer@mad.scientist.com) and GeNeura Team, 1999, 2000
 
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

#ifndef eoBackInserter_h
#define eoBackInserter_h

#include <eoInserter.h>

/**
\ingroup inserters
 * eoBackInserter: Interface class that enables an operator to insert
    new individuals at the back of the new population.
*/
template <class EOT>
class eoBackInserter : public eoPopInserter<EOT> 
{
    public :

    eoBackInserter(void) : eoPopInserter<EOT>() {}
        
    eoInserter<EOT>& operator()(const EOT& _eot)
    {
        pop().push_back(_eot);
        return *this;
    }

    string className(void) const { return "eoBackInserter"; }

};

#endif
