// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoPrintable.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef EOPRINTABLE_H
#define EOPRINTABLE_H

//-----------------------------------------------------------------------------

#include <iostream>  // istream, ostream
#include <string> // para string

using namespace std;

//-----------------------------------------------------------------------------
// eoPrintable
//-----------------------------------------------------------------------------
/**
Base class for objects that can print themselves
(#printOn#). Besides, this file defines the standard output for all the objects; 
if the objects define printOn there's no need to define #operator <<#.\\
This functionality was separated from eoObject, since it makes no sense to print
some objects (for instance, a #eoFactory# or a random number generator.
 */
class eoPrintable
{
 public:
  /// Virtual dtor. They are needed in virtual class hierarchies.
  virtual ~eoPrintable() {}
  
  /**
   * Write object. It's called printOn since it prints the object on a stream.
   * @param _os A ostream.
   */
  virtual void printOn(ostream& _os) const = 0;
};

//-----------------------------------------------------------------------------
///Standard output for all objects in the EO hierarchy
ostream & operator << ( ostream& _os, const eoPrintable& _o );

#endif EOPRINTABLE_H
