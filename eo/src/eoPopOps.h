// eoPopOps.h
// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eo1d.h 
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

#ifndef _EOPOPOPS_H
#define _EOPOPOPS_H

using namespace std;

/**
@author Geneura Team
@version 0.0
*/

//-----------------------------------------------------------------------------
#include <eoPop.h>

//-----------------------------------------------------------------------------
/** eoTransform is a class that transforms or does something on a population.
 */
template<class EOT>
class eoTransform: public eoObject{

 public:
  /** ctor */
  eoTransform() {};	
  
  /// Dtor
  virtual ~eoTransform(){};
  
  /// Pure virtual transformation function. Does something on the population
  virtual void operator () ( eoPop<EOT>& _pop ) = 0;
  
  /** @name Methods from eoObject	*/
  //@{
  /** readFrom and printOn are not overriden
   */
  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  string className() const {return "eoTransform";};
  //@}
  
};

//-----------------------------------------------------------------------------

/** eoSelect usually takes elements from one population, with or without transformation, and transfers them to the other population */
template<class EOT>
class eoSelect: public eoObject{

 public:
  /** ctor
   */
  eoSelect() {};	

  /// Dtor
  virtual ~eoSelect(){};

  /** Pure virtual transformation function. Extracts something from the parents,
      and transfers it to the siblings
      @param _parents the initial generation. Will be kept constant
      @param _siblings the created offspring. Will be usually an empty population
  */
  virtual void operator () ( const eoPop<EOT>& _parents, eoPop<EOT>& _siblings ) const = 0;

  /** @name Methods from eoObject	*/
  //@{
  /** readFrom and printOn are not overriden
   */
  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  string className() const {return "eoSelect";};
  //@}

};

/** eoMerge involves three populations, that can be merged and transformed to
give a third
*/
template<class EOT>
class eoMerge: public eoObject{

 public:
  /// (Default) Constructor.
  eoMerge(const float& _rate = 1.0): rep_rate(_rate) {}
  
  /// Dtor
  virtual ~eoMerge() {}
  
  /** Pure virtual transformation function. Extracts something from breeders
   *  and transfers it to the pop
   *  @param breeders Tranformed individuals.
   *  @param pop The original population at the begining, the result at the end
   */
  virtual void operator () ( eoPop<EOT>& breeders, eoPop<EOT>& pop ) = 0;
  
  /** @name Methods from eoObject       */
  //@{
  /** readFrom and printOn are not overriden
   */
  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  string className() const {return "eoMerge";};
  //@}
  
  /// Return the rate to be selected from the original population
  float rate() const { return rep_rate; }

  /// Set the rate to be obtained after replacement.
  /// @param _rate The rate.
  void rate(const float& _rate) { rep_rate = _rate; }
  
 private:
  float rep_rate;
};

//-----------------------------------------------------------------------------

#endif
