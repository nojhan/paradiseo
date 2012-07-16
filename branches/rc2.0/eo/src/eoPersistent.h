// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-



//-----------------------------------------------------------------------------

// eoPersistent.h
// (c) GeNeura Team, 1999
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

#ifndef EOPERSISTENT_H
#define EOPERSISTENT_H


#include <iostream>  // std::istream, std::ostream
#include <string>    // para std::string

#include "eoPrintable.h"

/**
 @addtogroup Core
 @{
*/

/**
An persistent object that knows how to write (through functions inherited from
#eoPrintable#) and read itself
 */
class eoPersistent: public eoPrintable
{
 public:
  /// Virtual dtor. They are needed in virtual class hierarchies.
  virtual ~eoPersistent() {}

  /**
   * Read object.
   * @param _is A std::istream.
   * @throw runtime_std::exception If a valid object can't be read.
   */
  virtual void readFrom(std::istream& _is) = 0;

};

///Standard input for all objects in the EO hierarchy
std::istream & operator >> ( std::istream& _is, eoPersistent& _o );

#endif

/** @} */
