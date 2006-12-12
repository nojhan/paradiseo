// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-



//-----------------------------------------------------------------------------
// eoRandomSelect.h

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

#ifndef EORANDOMSELECT_H
#define EORANDOMSELECT_H

//-----------------------------------------------------------------------------

#include <algorithm>
#include <numeric>    // for accumulate

#include <functional>

#include <eoPopOps.h>
#include <utils/eoRNG.h>

//-----------------------------------------------------------------------------

/** 
 * eoRandomSelect: an selection operator, which selects randomly a percentage

 of the initial population.
 */
template<class EOT> class eoRandomSelect: public eoBinPopOp<EOT>
{
 public:
  ///
  eoRandomSelect(const float& _percent = 0.4): 
    eoBinPopOp<EOT>(), repRate(_percent) {};
  
  ///
  virtual ~eoRandomSelect() {};
  
  /// Takes a percentage of the population randomly, and transfers it to siblings
  virtual void operator() ( eoPop<EOT>& _parents, eoPop<EOT>& _siblings )  {
    // generates random numbers
    unsigned num_chroms = (unsigned)(repRate * _parents.size());

    // selection of chromosomes
    do {
      _siblings.push_back(_parents[rng.random(_parents.size())]);
    } while (_siblings.size() < num_chroms);
  }



    /// @name Methods from eoObject

  //@{

  /**

   * Read object. Reads the percentage

   * Should call base class, just in case.

   * @param _s A std::istream.

   */

  virtual void readFrom(std::istream& _s) {

	_s >> repRate;

  }



  /** Print itself: inherited from eoObject implementation. Declared virtual so that 

      it can be reimplemented anywhere. Instance from base classes are processed in

	  base classes, so you don´t have to worry about, for instance, fitness.

  @param _s the std::ostream in which things are written*/

  virtual void printOn( std::ostream& _s ) const{

	_s << repRate;

  }



  /** Inherited from eoObject 

      @see eoObject

  */

  std::string className() const {return "eoRandomSelect";};



  //@}




 private:
  float repRate;
};

//-----------------------------------------------------------------------------

#endif EOGSRANDOMSELECT_H
