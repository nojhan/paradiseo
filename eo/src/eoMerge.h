// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoMerge.h
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

#ifndef eoMerge_h
#define eoMerge_h

//-----------------------------------------------------------------------------

#include <iostream>

// EO includes
#include <eoPop.h>     // eoPop
#include <eoPopOps.h>  // eoMerge

/**
 * eoMerge: Base class for replacement algorithms
 */

template<class Chrom> class eoMerge: public eoBinPopOp<Chrom>
{
 public:
  /// (Default) Constructor.
  eoMerge(const float& _rate = 1.0): eoBinPopOp<Chrom>(), repRate( _rate ) {}

  /// Ctor from istream
  eoMerge( istream& _is): eoBinPopOp<Chrom>() { readFrom( _is ); };
 
  /**
   * Creates a new population based on breeders and original populations.
   * @param breeders The population of breeders. Should be sorted to work correctly
   * @param pop The original population.
   */
  void operator()( eoPop<Chrom>& _breeders, eoPop<Chrom>& _pop)
    {
      unsigned target = static_cast<unsigned>(rint(_pop.size() * rate()));
      
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
  /**
   * Read object. The EOT class must have a ctor from a stream;
   in this case, a strstream is used.
   * @param _is A istream.
   
   */
  virtual void readFrom(istream& _is) {
    _is >> repRate;
  }
  
  /**
   * Write object. Prints relevant parameters to standard output
   * @param _os A ostream. In this case, prints the population to
   standard output. The EOT class must hav standard output with cout,
   but since it should be an eoObject anyways, it's no big deal.
   */
  virtual void printOn(ostream& _os) const {
    _os << repRate;
  };

  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  virtual string className() const {return "eoMerge";};
  //@}

 protected:
  float rate() { return repRate;};

 private:
  float repRate;

};

//-----------------------------------------------------------------------------

#endif eoInsertion_h
