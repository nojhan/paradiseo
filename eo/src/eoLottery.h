// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoLottery.h
//   Implements the lottery procedure for selection
// (c) GeNeura Team, 1998 - Marc Schoenauer, 2000
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

#ifndef eoLottery_h
#define eoLottery_h

//-----------------------------------------------------------------------------

#include <functional>  // 
#include <numeric>     // accumulate
#include "selectors.h"
#include <eo>          // eoPop eoSelect MINFLOAT

//-----------------------------------------------------------------------------
/** eoLottery: a selection method. Puts into the output a group of individuals 
    selected using lottery; individuals with higher probability will have more
    chances of being selected.
    Requires EOT::Fitness to be float castable
*/
//-----------------------------------------------------------------------------

template<class EOT> class eoLottery: public eoBinPopOp<EOT>
{
 public:
  /// (Default) Constructor.
  eoLottery(const float& _rate = 1.0): eoBinPopOp<EOT>(), rate(_rate) 
  {
      if (minimizing_fitness<EOT>())
      {
          eoMinimizingFitnessException up(*this);
          throw up; // :-)
      }
  }
  
  /** actually selects individuals from pop and pushes them back them into breeders
   *  until breeders has the right size: rate*pop.size()
   *  BUT what happens if breeders is already too big?
   * Too big for what?
   */
  void operator()( eoPop<EOT>& pop, eoPop<EOT>& breeders) 
    {
      int target = (int)(rate * pop.size());

      // test of consistency
      if (breeders.size() >= target) {
	  throw("Problem in eoLottery: already too many offspring");
      }

      double total;
      
      try
      {
          total = sum_fitness(pop);
      }
      catch (eoNegativeFitnessException&)
      { // say where it occured...
          throw eoNegativeFitnessException(*this);
      }

      // selection of chromosomes
      while (breeders.size() < target) 
      {
	    breeders.push_back(roulette_wheel(pop, total));
      }
    }

  double Rate(void) const { return rate; }
  
 private:
  double rate;  // selection rate
};

//-----------------------------------------------------------------------------

#endif eoLottery_h
