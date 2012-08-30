// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoPrintable.h
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

#ifndef EOPRINTABLE_H
#define EOPRINTABLE_H

//-----------------------------------------------------------------------------

#include <iostream>  // std::istream, std::ostream
#include <string> // para std::string

/*
This functionality was separated from eoObject, since it makes no sense to print
some objects (for instance, a #eoFactory# or a random number generator.
*/

/**
Base class for objects that can print themselves
(#printOn#). Besides, this file defines the standard output for all the objects;
if the objects define printOn there's no need to define "operator<<".

  @ingroup Core
 */
class eoPrintable
{
 public:
  /// Virtual dtor. They are needed in virtual class hierarchies.
  virtual ~eoPrintable() {}

  /**
   * Write object. It's called printOn since it prints the object on a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const = 0;
};

//-----------------------------------------------------------------------------
///Standard output for all objects in the EO hierarchy
std::ostream & operator << ( std::ostream& _os, const eoPrintable& _o );

#endif
