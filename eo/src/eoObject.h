// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoObject.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef EOOBJECT_H
#define EOOBJECT_H

//-----------------------------------------------------------------------------

#include <eoData.h>		// For limits definition
#include <iostream>		// istream, ostream
#include <string>		// para string

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
