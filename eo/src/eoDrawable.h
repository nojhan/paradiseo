// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoDrawable.h
// (c) GeNeura Team, 1999
//-----------------------------------------------------------------------------

#ifndef EODRAWABLE_H
#define EODRAWABLE_H

//-----------------------------------------------------------------------------

using namespace std;

//-----------------------------------------------------------------------------
// eoDrawable
//-----------------------------------------------------------------------------

/** eoDrawable is a template class that adds a drawing interface to an object.\\
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

#endif EODRAWABLE_H
