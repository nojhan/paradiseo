/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoRandomIndiSelector.h
    Selects individuals at random.

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

#ifndef eoRandomIndiSelector_h
#define eoRandomIndiSelector_h

#include "eoIndiSelector.h"

/**
\ingroup selectors
 * eoRandomSelector: just selects a random child
*/
template <class EOT>
class eoRandomIndiSelector : public eoPopIndiSelector<EOT>
{
    public :

    eoRandomIndiSelector(void) : eoPopIndiSelector<EOT>() {}
    virtual ~eoRandomIndiSelector(void) {}

    /// very complex function that returns just an individual
    const EOT& do_select(void) 
    {
        return operator[](rng.random(size()));
    }

};

#endif


