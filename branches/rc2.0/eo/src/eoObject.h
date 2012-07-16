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

#include <utils/eoData.h>	// For limits definition
#include <iostream>		// std::istream, std::ostream
#include <string>		// std::string

#include <utils/compatibility.h>

/*
eoObject used to be the base class for the whole hierarchy, but this has
changed. eoObject is used to define a name (#className#)
that is used when loading or saving the state.

Previously, this object also defined a print and read
interface, but it´s been moved to eoPrintable and eoPersistent.
*/

/** Defines a name (#className#), used when loading or saving a state.

It is recommended that you only derive from eoObject in concrete classes.
Some parts of EO do not implement this yet, but that will change in the future.
eoObject, together with eoPersistent and eoPrintable provide a simple persistence
framework that is only needed when the classes have state that changes at runtime.

  @see eoPersistent eoPrintable, eoState

  @ingroup Core
 */
class eoObject
{
 public:
  /// Virtual dtor. They are needed in virtual class hierarchies.
  virtual ~eoObject() {}

  /** Return the class id. This should be redefined in each class.
  Only "leaf" classes can be non-virtual.

  Maarten: removed the default implementation as this proved to
  be too error-prone: I found several classes that had a typo in
  className (like classname), which would print eoObject instead of
  their own. Having it pure will force the implementor to provide a
  name.
  */
  virtual std::string className() const = 0;

};

#endif
