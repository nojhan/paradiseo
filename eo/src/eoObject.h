// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoObject.h
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

#ifndef EOOBJECT_H
#define EOOBJECT_H

//-----------------------------------------------------------------------------

#include <eoData.h>		// For limits definition
#include <iostream>		// istream, ostream
#include <string>		// string

#include "compatibility.h"

using namespace std;

//-----------------------------------------------------------------------------
// eoObject
//-----------------------------------------------------------------------------
/**
This is the base class for the whole hierarchy; an eoObject defines
basically an interface for the whole hierarchy: each object should
know its name (#className#). Previously, this object defined a print and read
interface, but it´s been moved to eoPrintable and eoPersistent.
 */
class eoObject
{
 public:
  
  /// Default Constructor.
  eoObject() {}

  /// Copy constructor.
  eoObject( const eoObject& ) {}

  /// Virtual dtor. They are needed in virtual class hierarchies.
  virtual ~eoObject() {}
  
  /** Return the class id. This should be redefined in each class; but 
  it's got code as an example of implementation. Only "leaf" classes
  can be non-virtual.
  */
  virtual string className() const { return "eoObject"; }

};

#endif EOOBJECT_H

