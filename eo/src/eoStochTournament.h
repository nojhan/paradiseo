// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoStochTournament.h
// (c) GeNeura Team, 1998 - EEAAX 1999
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
             Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

#ifndef eoStochTournament_h
#define eoStochTournament_h

//-----------------------------------------------------------------------------

#include <functional>  // 
#include <numeric>     // accumulate
#include <eoPopOps.h>          // eoPop eoSelect MINFLOAT
#include <eoRNG.h>

//-----------------------------------------------------------------------------
/** eoStochTournament: a selection method that selects ONE individual by
 binary stochastic tournament 
 -MS- 24/10/99 */
//-----------------------------------------------------------------------------

template <class EOT> class eoStochTournament: public eoSelectOne<EOT>
{
 public:

  ///
  eoStochTournament(float _Trate = 1.0 ):eoSelectOne(), Trate(_Trate) {
    // consistency check
    if (Trate < 0.5) {
      cerr << "Warning, Tournament rate should be > 0.5\nAdjusted to 0.55\n";
      Trate = 0.55;
    }
  }
  
  /** DANGER: if you want to be able to minimize as well as maximizem
      DON'T cast the fitness to a float, use the EOT comparator! */
  virtual const EOT& operator()(const eoPop<EOT>& pop) {
    unsigned i1 = rng.random(pop.size()),
      i2 = rng.random(pop.size());

    bool ok = ( rng.flip(Trate) );
    if (pop[i1] < pop[ i2 ] ) {
      if (ok) return pop[ i2 ];
      else    return pop[ i1 ];
    }
    else {
      if (ok) return pop[ i1 ];
      else    return pop[ i2 ];
    }
  }
  
private:
  float Trate;
};

//-----------------------------------------------------------------------------

#endif eoDetTournament_h
