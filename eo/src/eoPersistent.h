// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoPersistent.h
// (c) GeNeura Team, 1999
//-----------------------------------------------------------------------------

#ifndef EOPERSISTENT_H
#define EOPERSISTENT_H

/// @name variables Some definitions of variables used throughout the program
//@{
/// max length to store stuff read
const unsigned MAXLINELENGTH=1024;
//@}

//-----------------------------------------------------------------------------

#include <iostream>  // istream, ostream
#include <string> // para string

//-----------------------------------------------------------------------------
#include <eoPrintable.h>

using namespace std;

//-----------------------------------------------------------------------------
// eoPersistent
//-----------------------------------------------------------------------------
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
   * @param _is A istream.
   * @throw runtime_exception If a valid object can't be read.
   */
  virtual void readFrom(istream& _is) = 0;
  
};

//-----------------------------------------------------------------------------
///Standard input for all objects in the EO hierarchy
istream & operator >> ( istream& _is, eoPersistent& _o );

#endif EOOBJECT_H
