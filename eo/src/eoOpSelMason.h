// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoOpSelMason.h
// (c) GeNeura Team, 1999
//-----------------------------------------------------------------------------

#ifndef _EOOPSELMASON_H
#define _EOOPSELMASON_H

//-----------------------------------------------------------------------------
#include <eoProportionalOpSel.h>
#include <eoOpFactory.h>	// for eoFactory and eoOpFactory

#include <map>

//-----------------------------------------------------------------------------

/** EO Mason, or builder, for operator selectors. A builder must allocate memory
to the objects it builds, and then deallocate it when it gets out of scope*/
template<class eoClass>
class eoOpSelMason: public eoFactory<eoOpSelector<eoClass> > {
	
public:
	typedef vector<eoOp<eoClass>* > vOpP;
	typedef map<eoOpSelector<eoClass>*, vOpP > MEV;

	/// @name ctors and dtors
	//{@
	/// constructor
	eoOpSelMason( eoOpFactory<eoClass>& _opFact): operatorFactory( _opFact ) {};
	
	/// destructor
	virtual ~eoOpSelMason() {};
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
		eoOpSelector<eoClass>* opSelectorP;
		// Build the operator selector
		if ( opSelName == "eoProportionalOpSel" ) {
			opSelectorP = new eoProportionalOpSel<eoClass>();
		}

		// Temp vector for storing pointers
		vOpP tmpPVec;
		// read operator rate and name
		while ( _is ) {
			float rate;
			_is >> rate;
			if ( _is ) {
				eoOp<eoClass>* op = operatorFactory.make( _is );	// This reads the rest of the line
				// Add the operators to the selector, don´t pay attention to the IDs
				opSelectorP->addOp( *op, rate );
				// Keep it in the store, to destroy later
				tmpPVec.push_back( op );
			} // if
		} // while

		// Put it in the map
		allocMap.insert( MEV::value_type( opSelectorP, tmpPVec ) );
		
		return opSelectorP;
	};

	///@name eoObject methods
	//@{
	/** Return the class id */
	virtual string className() const { return "eoOpSelMason"; }

	//@}
	
private:
	map<eoOpSelector<eoClass>*,vector<eoOp<eoClass>* > > allocMap;
	eoOpFactory<eoClass>& operatorFactory;
};


#endif _EOOPSELMASON_H
