// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoDrawable.h
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

#ifndef EODRAWABLE_H
#define EODRAWABLE_H

//-----------------------------------------------------------------------------

using namespace std;

//-----------------------------------------------------------------------------
// eoDrawable
//-----------------------------------------------------------------------------

/** eoDrawable is a template class that adds a drawing interface to an object.
Requisites for template instantiation are that the object must admit a default ctor
and a copy ctor. The Object must be an eoObject, thus, it must have its methods: className,
eoDrawables can be drawn on any two-dimensional surface; it can be added to any
object with above characteristics.
@see eoObject
*/
template <class Object>
class eoDrawable
{
 public:
	/// Main ctor from an already built Object.
	eoDrawable( const Object& _o): Object( _o ){};

	/// Copy constructor.
	eoDrawable( const eoDrawable& _d): Object( _d ){};

  /// Virtual dtor. They are needed in virtual class hierarchies
  virtual ~eoDrawable() {};


  /**Draws the object. It must be redefined in any subclass, it´s impossible
  to have a general drawing method
  @param _x, _y coorinates */
  virtual void draw( unsigned _x, unsigned _y) = 0;

};

#endif //! EODRAWABLE_H
