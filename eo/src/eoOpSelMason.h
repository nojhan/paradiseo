// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoOpSelMason.h
// (c) GeNeura Team, 1999
//-----------------------------------------------------------------------------

#ifndef _EOOPSELMASON_H
#define _EOOPSELMASON_H

//-----------------------------------------------------------------------------
#include <eoOpSelector.h>

//-----------------------------------------------------------------------------

/** EO Mason, or builder, for operator selectors. A builder must allocate memory
to the objects it builds, and then deallocate it when it gets out of scope*/
template<class eoClass>
class eoOpSelMason: public eoFactory<eoOpSelector<eoClass> > {
	
public:
	
	/// @name ctors and dtors
	//{@
	/// constructor
	eoOpSelMason( ) {}
	
	/// destructor
	virtual ~eoOpSelMason() {}
	//@}

	/** Factory methods: creates an object from an istream, reading from
	it whatever is needed to create the object. The format is
	opSelClassName\\
	rate 1 operator1\\
	rate 2 operator2\\
	...\\
	Stores all operators built in a database (#allocMap#), so that somebody 
	can destroy them later. The Mason is in charge or destroying the operators,
	since the built object can´t do it itself. The objects built must be destroyed
	from outside, using the #destroy# method
	*/
	virtual eoOpSelector<eoClass>* make(istream& _is) {
		string opSelName;
		_is >> opSelName;
		eoMonOpFactory<eoClass> selMaker;
		// read operator rate and name
		while ( _is ) {
			float rate;
			_is >> rate;
	

			// Create an stream
			strstream s0;
			eoMonOp<IEO>* op0 = selMaker.make( s0 );
		}
	
	}

	///@name eoObject methods
	//@{
	/** Return the class id */
	virtual string className() const { return "eoFactory"; }

	/** Read and print are left without implementation */
	//@}
	
private:
	map<eoOpSelector<eoClass>*,vector<eoOp<eoClass>* > > allocMap;
};


#endif _EOFACTORY_H
