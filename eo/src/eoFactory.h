// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFactory.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EOFACTORY_H
#define _EOFACTORY_H

//-----------------------------------------------------------------------------
#include <eoObject.h>

//-----------------------------------------------------------------------------

/** EO Factory. A factory is used to create other objects. In particular,
it can be used so that objects of that kind can´t be created in any other
way. It should be instantiated with anything that needs a factory, like selectors
or whatever; but the instance class should be the parent class from which all the
object that are going to be created descend. This class basically defines an interface,
as usual. The base factory class for each hierarchy should be redefined every time a new
object is added to the hierarchy, which is not too good, but in any case, some code would
have to be modified*/
template<class EOClass>
class eoFactory: public eoObject {
	
public:
	
	/// @name ctors and dtors
	//{@
	/// constructor
	eoFactory( ) {}
	
	/// destructor
	virtual ~eoFactory() {}
	//@}

	/** Another factory methods: creates an object from an istream, reading from
	it whatever is needed to create the object. Usually, the format for the istream will be\\
	objectType parameter1 parameter2 ... parametern\\
	*/
	virtual EOClass* make(istream& _is) = 0;

	///@name eoObject methods
	//@{
	/** Return the class id */
	virtual string className() const { return "eoFactory"; }

	/** Read and print are left without implementation */
	//@}
	
};


#endif _EOFACTORY_H
