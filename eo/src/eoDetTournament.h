// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoDetTournament.h
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

#ifndef eoDetTournament_h
#define eoDetTournament_h

//-----------------------------------------------------------------------------

#include <functional>  // 
#include <numeric>     // accumulate
#include <eoPopOps.h>          // eoPop eoSelect MINFLOAT
#include <eoRNG.h>

//-----------------------------------------------------------------------------
/** eoDetTournament: a selection method that selects ONE individual by
 deterministic tournament 
 -MS- 24/10/99 */
//-----------------------------------------------------------------------------

template <class EOT> class eoDetTournament: public eoSelectOne<EOT>
{
 public:
  /// (Default) Constructor.
  eoDetTournament(unsigned _Tsize = 2 ):eoSelectOne(), Tsize(_Tsize) {
    // consistency check
    if (Tsize < 2) {
      cout << "Warning, Tournament size shoudl be >= 2\nAdjusted\n";
      Tsize = 2;
    }
  }
  
  /** DANGER: if you want to be able to minimize as well as maximizem
      DON'T cast the fitness to a float, use the EOT comparator! */
  virtual const EOT& operator()(const eoPop<EOT>& pop) {
    unsigned best = rng.random(pop.size());		// random individual
    
    for (unsigned i = 0; i<Tsize-1; i++) {
      unsigned tmp = rng.random(pop.size());
      if (pop[best] < pop[tmp])
	best = tmp;
    }
    return pop[best] ;
  }

  /** Inherited from eoObject 
      @see eoObject
  */
  string className() const {return "eoDetTournament";};
  
 private:
  unsigned Tsize;
};

//-----------------------------------------------------------------------------

#endif eoDetTournament_h
