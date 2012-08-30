// eoAged.h
// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoAge.h
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

#ifndef EOAGED_H
#define EOAGED_H

//-----------------------------------------------------------------------------

#include <iostream>  // istream, ostream
#include <std::string> // para std::string

using namespace std;

//-----------------------------------------------------------------------------
// eoAge
//-----------------------------------------------------------------------------

/** eoAge is a template class that adds an age to an object.\\
Requisites for template instantiation are that the object must admit a default ctor
and a copy ctor. The Object must be an eoObject, thus, it must have its methods: className,
printOn, readFrom.
@see eoObject
*/
template <class Object>
class eoAged: public Object
{
 public:
	/// Main ctor from an already built Object.
	eoAged( const Object& _o): Object( _o ), age(0) {};

  /// Copy constructor.
  eoAged( const eoAged& _a): Object( _a ), age( _a.age ) {};

  /// Virtual dtor. They are needed in virtual class hierarchies
  virtual ~eoAged() {};


  ///returns the age of the object
  unsigned long Age() const {return age;}

  /// Increments age
  const eoAged& operator ++ () { age++; return *this;}

  	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eo1d
	*/
//@{
  /** Return the class id. This should be redefined in each class; but
  it's got code as an example of implementation. Only "leaf" classes
  can be non-virtual.
  */
  virtual std::string className() const { return std::string("eoAged")+Object::className(); };

  /**
   * Read object.
   * @param _is A istream.
   * @throw runtime_exception If a valid object can't be read.
   */
  virtual void readFrom(istream& _is) {
	  Object::readFrom( _is );
	  _is >> age;
  }


  /**
   * Write object. It's called printOn since it prints the object _on_ a stream.
   * @param _os A ostream.
   */
  virtual void printOn(ostream& _os) const{
	  Object::printOn( _os );
	  _os << age;
  }
//@}

 private:

	 /** Default Constructor. \\
	 It´s private so that it is not used anywhere; the right way of using this object
	 is to create an Object and passing it to an aged by means of the copy ctor; that way
	 it´s turned into an Aged object*/
	 eoAged(): Object(), age(0) {};

	 unsigned long age;
};

#endif EOAGE_H
