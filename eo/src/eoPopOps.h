/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoPopOps.h 
   (c) GeNeura Team, 1998
 
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

@version 0.1 : -MS- 22/10/99
    added the added the derived class eoSelectOne for which you only have
          to define the selection of 1 individual 
          (e.g. for tournament, age selection in SSGA, ...)
    added at the BASE level (after all it's themost frequen case)
          the pure virtual operator that selects one single individual:
          EOT eoSelect::operator ( const eoPop<EOT>& _parents, 
				        const EOT& _first = 0)
    added the optional second parameter to transform::operator()
*/

//-----------------------------------------------------------------------------
#include <eoPop.h>

//-----------------------------------------------------------------------------
/** eoTransform is a class that transforms or does something on a population.
 */
template<class EOT>
class eoMonPopOp: public eoObject{

 public:
  /** ctor */
  eoMonPopOp() {};	
  
  /// Dtor
  virtual ~eoMonPopOp(){};
  
  /// Pure virtual transformation function. Does something on the population
  virtual void operator () ( eoPop<EOT>& _pop ) = 0;
  
  /** @name Methods from eoObject	*/
  //@{
  /** readFrom and printOn are not overriden
   */
  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  virtual string className() const {return "eoMonPopOp";};
  //@}
  
};

//-----------------------------------------------------------------------------

/** eoSelect usually takes elements from one population, with or without transformation, and transfers them to the other population */
template<class EOT>
class eoBinPopOp: public eoObject{

 public:
  /** ctor
   */
  eoBinPopOp() {};	

  /// Dtor
  virtual ~eoBinPopOp(){};

  /** Pure virtual transformation function. Extracts something from the parents,
      and transfers it to the siblings
      @param _parents the initial generation. Will be kept constant
      @param _siblings the created offspring. Will be usually an empty population
  */
  virtual void operator () (const eoPop<EOT>& _parents, 
			          eoPop<EOT>& _siblings ) = 0;

  /** @name Methods from eoObject	*/
  //@{
  /** readFrom and printOn are not overriden
   */
  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  virtual string className() const {return "eoBinPopOp";};
  //@}

};

//-----------------------------------------------------------------------------

/** eoSelectone selects only one element from a whole population. Usually used to 
    select mates*/
template<class EOT>
class eoSelectOne: public eoObject{

 public:
  /** ctor
   */
  eoSelectOne() {};	

  /// Dtor
  virtual ~eoSelectOne(){};

  /** Pure virtual transformation function. Extracts something from the parents,
      and transfers it to the siblings
      @param _parents the initial generation. Will be kept constant
      @param _siblings the created offspring. Will be usually an empty population
  */
  virtual const EOT& operator () ( const eoPop<EOT>& _parents ) = 0;

  /** @name Methods from eoObject	*/
  //@{
  /** readFrom and printOn are not overriden
   */
  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  virtual string className() const {return "eoSelectOne";};
  //@}

};
#endif
