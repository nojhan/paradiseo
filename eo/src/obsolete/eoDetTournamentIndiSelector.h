/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoDetTournamentIndiSelector.h

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

#ifndef eoDetTournamentIndiSelector_h
#define eoDetTournamentIndiSelector_h

#include "eoIndiSelector.h"
#include "utils/selectors.h"


/**
\ingroup selectors
  * eoDetTournamentIndiSelector: selects children through a deterministic_tournament
*/
template <class EOT>
class eoDetTournamentIndiSelector : public eoPopIndiSelector<EOT>
{
    public :

    eoDetTournamentIndiSelector(int _tournamentSize) 
            : eoPopIndiSelector<EOT>(),
              tournamentSize(_tournamentSize) 
              {}
    
    virtual ~eoDetTournamentIndiSelector(void) {}

    const EOT& do_select(void) 
    {
        return *deterministic_tournament(begin(), end(), tournamentSize);
    }

    private :

        int tournamentSize;
};

#endif
