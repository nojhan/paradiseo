// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoInsertion.h
//   Inserts new members into the population
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

#ifndef eoInsertion_h
#define eoInsertion_h

//-----------------------------------------------------------------------------

#include <iostream>

// EO includes
#include <eoPop.h>     // eoPop
#include <eoMerge.h>  // eoMerge

/******************************************************************************
 * eoInsertion: A replacement algorithm.
 * Creates a new population with all the breeders and the best individuals 
 * from the original population.
 *****************************************************************************/

template<class Chrom> class eoInsertion: public eoMerge<Chrom>
{
 public:
  /// (Default) Constructor.
  eoInsertion(const float& _rate = 1.0): eoMerge<Chrom>( _rate ) {}

  /// Ctor from istream
  eoInsertion( istream& _is): eoBinPopOp<Chrom>( _is ) {};

  /// Dtor
  virtual ~eoInsertion() {};

  /**
   * Creates a new population based on breeders and original populations.
   * @param breeders The population of breeders. Should be sorted to work correctly
   * @param pop The original population.
   */
  void operator()( eoPop<Chrom>& _breeders, eoPop<Chrom>& _pop)
    {
      unsigned target = static_cast<unsigned>((_pop.size() * rate()));
      
      _pop.swap(_breeders);
      
      if (target < _pop.size())
	{
	  partial_sort(_pop.begin(), _pop.begin() + target, _pop.end(), 
		       greater<Chrom>());
	  _pop.erase(_pop.begin() + target, _pop.end());
	}
      else
	{
	  target = min(target - _pop.size(), _breeders.size());
	  partial_sort(_breeders.begin(), _breeders.begin() + target, 
		       _breeders.end(), greater<Chrom>());
	  copy(_breeders.begin(), _breeders.begin() + target,
	       back_insert_iterator<eoPop<Chrom> >(_pop));
	}
    };

  /** @name Methods from eoObject	*/
  //@{
  /** readFrom and printOn inherited from eoMerge */

  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  virtual string className() const {return "eoInsertion";};
  //@}

};

//-----------------------------------------------------------------------------

#endif eoInsertion_h
