/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoStochTournamentInserter.h
    Concrete steady state inserter. It is initialized with a population and 
    inserts individuals in the population based on an inverse stochastic
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

#ifndef eoStochTournamentInserter_h
#define eoStochTournamentInserter_h


#include <eoSteadyStateInserter.h>
#include <utils/selectors.h>

/**
\ingroup inserters
 * eoStochTournamentInserter: Uses an inverse stochastic tournament to figure
 * out who gets overridden by the new individual. It resets the fitness of the
 * individual.
*/
template <class EOT>
class eoStochTournamentInserter : public eoSteadyStateInserter<EOT> 
{
public :
  
  eoStochTournamentInserter(eoEvalFunc<EOT>& _eval, double _t_rate): 
    eoSteadyStateInserter<EOT>(_eval), t_rate(_t_rate)
  {
    if (t_rate < 0.5)
      { // warning, error?
	t_rate = 0.55;
      }

    if (t_rate >= 1.0)
      {
	t_rate = 0.99; // 1.0 would mean deterministic tournament
      }
  }
        
  eoInserter<EOT>& operator()(const EOT& _eot)
  {
    EOT& eo = inverse_stochastic_tournament<EOT>(pop(), t_rate);
    eo = _eot; // overwrite loser of tournament
        
    eo.invalidate();
    return *this;
  }

  string className(void) const { return "eoStochTournamentInserter"; }

private :
  double t_rate;
};

#endif
