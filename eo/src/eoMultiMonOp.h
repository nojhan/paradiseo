// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoMultiMonOp.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EOMULTIMONOP_h
#define _EOMULTIMONOP_h

#include <iterator>

#include <eoOp.h>

/** MultiMonOp combines several monary operators. By itself, it does nothing to the
EO it´s handled*/
template <class EOT>
class eoMultiMonOp: public eoMonOp<EOT>  {
public:
	/// Ctor from an already existing op
	eoMultiMonOp( const eoMonOp<EOT>* _op )
		: eoMonOp< EOT >( ), vOp(){
		vOp.push_back( _op );
	};

	///
	eoMultiMonOp( )
		: eoMonOp< EOT >( ), vOp(){};

	/// Ctor from an already existing op
	void adOp( const eoMonOp<EOT>* _op ){
		vOp.push_back( _op );
	};

	/// needed virtual dtor
	virtual ~eoMultiMonOp() {};

	///
	/// Applies all operators to the EO
	virtual void operator()( EOT& _eo ) const {
		if ( vOp.begin() != vOp.end() ) {
			for ( vector<const eoMonOp<EOT>* >::const_iterator i = vOp.begin(); 
					i != vOp.end(); i++ ) {
				(*i)->operator () ( _eo );
			}
		}
	}


	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eoOp
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	string className() const {return "eoMonOp";};
    //@}
private:
	vector< const eoMonOp<EOT>* > vOp;
};

#endif