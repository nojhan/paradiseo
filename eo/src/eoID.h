// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoID.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef EOID_H
#define EOID_H

//-----------------------------------------------------------------------------

#include <iostream>  // istream, ostream
#include <string> // for string

using namespace std;

//-----------------------------------------------------------------------------
// eoID
//-----------------------------------------------------------------------------

/** eoID is a template class that adds an ID to an object.\\
Requisites for template instantiation are that the object must admit a default ctor 
and a copy ctor. The Object must be an eoObject, thus, it must have its methods: className,
printOn, readFrom, that is why we don´t subclass eoObject to avoid multiple inheritance.\\
IDs can be used to count objects of a a kind, or track them, or whatever.
@see eoObject
*/
template <class Object>
class eoID: public Object
{
 public:
	/// Main ctor from an already built Object.
	eoID( const Object& _o): Object( _o ), thisID(globalID++) {};

	/// Copy constructor.
	eoID( const eoID& _id): Object( _id ), thisID(globalID++ ) {};

  /// Virtual dtor. They are needed in virtual class hierarchies
  virtual ~eoID() {};
  

  ///returns the age of the object
  unsigned long ID() const {return thisID;}

  	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eo1d
	*/
	//@{
  /** Return the class id. This should be redefined in each class; but 
  it's got code as an example of implementation. Only "leaf" classes
  can be non-virtual.
  */
  virtual string className() const { return string("[eoID]")+Object::className(); };

  /**
   * Read object.
   * @param _is A istream.
   * @throw runtime_exception If a valid object can't be read.
   */
  virtual void readFrom(istream& _is) {
	  Object::readFrom( _is );
	  _is >> thisID;
  }

  
  /**
   * Write object. It's called printOn since it prints the object _on_ a stream.
   * @param _os A ostream.
   */
  virtual void printOn(ostream& _os) const{
	  Object::printOn( _os );
	  _os << thisID;
  }
//@}
  
 private:

	 /** Default Constructor. \\
	 It´s private so that it is not used anywhere; the right way of using this object
	 is to create an Object and passing it to an aged by means of the copy ctor; that way
	 it´s turned into an Aged object*/
	 eoID(): Object(), thisID( globalID++ ) {};

	 unsigned long thisID;
	 static unsigned long globalID;
};

template< class Object>
unsigned long eoID< Object >::globalID = 0;

#endif EOID_H
