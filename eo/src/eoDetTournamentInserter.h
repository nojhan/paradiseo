/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoDetTournamentInserter.h
    Concrete steady state inserter. It is initialized with a population and 
    inserts individuals in the population based on an inverse deterministic 
    tournament

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

#ifndef eoDetTournamentInserter_h
#define eoDetTournamentInserter_h


#include "eoSteadyStateInserter.h"
#include "selectors.h"

/**
 * eoDetTournamentInserter: Uses an inverse deterministic tournament to figure
 * out who gets overridden by the new individual. It resets the fitness of the
 * individual.
*/
template <class EOT>
class eoDetTournamentInserter : public eoSteadyStateInserter<EOT> 
{
    public :

    eoDetTournamentInserter(eoEvalFunc<EOT>& _eval, unsigned _t_size) : t_size(_t_size), eoSteadyStateInserter<EOT>(_eval)
    {
        if (t_size < 2)
        { // warning, error?
            t_size = 2;
        }
    }
        
    void insert(const EOT& _eot)
    {
        EOT& eo = inverse_deterministic_tournament<EOT>(pop(), t_size);
        eo = _eot; // overwrite loser of tournament

        eo.invalidate(); // This line should probably be removed when all genetic operators do this themselves
        eval(eo); // Evaluate after insert
    }

    string className(void) const { return "eoDetTournamentInserter"; }

    private :

        unsigned t_size;
};

#endif