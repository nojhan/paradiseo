/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    -----------------------------------------------------------------------------
    eoOpSelector.h
      Base class for operator selectors, which return 1 operator according
      to some criterium

    (c) GeNeura Team 1998, 1999, 2000
 
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

#ifndef EOOPSELECTOR_H
#define EOOPSELECTOR_H

//-----------------------------------------------------------------------------

#include <stdexcept>  // runtime_error 

#include <eoObject.h>
#include <eoPrintable.h>
#include <eoOp.h>

//-----------------------------------------------------------------------------
/** An operator selector is an object that contains a set of EO operators,
and selects one based on whatever criteria. It will be used in the breeder objects.\\
This class is basically a generic interface for operator selection
*/
template<class EOT>
class eoOpSelector: public eoObject, public eoPrintable
{
public:
  
    // Need virtual destructor for derived classes
    virtual ~eoOpSelector() {}

  /// type of IDs assigned to each operators, used to handle them
  typedef unsigned ID;
  
  /** add an operator to the operator set
      @param _op a genetic operator, that will be applied in some way
      @param _arg the operator rate, usually, or any other argument to the operator
      @return an ID that will be used to identify the operator
  */
  virtual ID addOp( eoOp<EOT>& _op, float _arg ) = 0;
  
  /** Gets a non-const reference to an operator, so that it can be changed, 
      modified or whatever 
      @param _id  a previously assigned ID
      @throw runtime_exception if the ID does not exist*/
  virtual eoOp<EOT>& getOp( ID _id ) = 0;
  
  /** Remove an operator from the operator set
      @param _id a previously assigned ID
      @throw runtime_exception if the ID does not exist
  */
  virtual void deleteOp( ID _id ) = 0;
  
  /// Returns a genetic operator according to the established criteria
  virtual eoOp<EOT>* Op() = 0;
  
  /// Methods inherited from eoObject
  //@{
  
  /** Return the class id. 
      @return the class name as a string
  */
  virtual string className() const { return "eoOpSelector"; };
  
  /**
   * Read object and print objects are left for subclasses to define.
   */
  //@}
};

//-----------------------------------------------------------------------------

#endif EO_H
