// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoKill.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EOKILL_h
#define _EOKILL_h

#include <eoUniform.h>

#include <eoOp.h>

/// Kill eliminates a gen in a chromosome
template <class EOT >
class eoKill: public eoMonOp<EOT>  {
public:
	///
	eoKill( )
		: eoMonOp< EOT >(){};

	/// needed virtual dtor
	virtual ~eoKill() {};

	///
	virtual void operator()( EOT& _eo ) const {
	  eoUniform<unsigned> uniform( 0, _eo.length() );
	  unsigned pos = uniform( );
	  _eo.deleteGene( pos );
	}

	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eoOp
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	virtual string className() const {return "eoKill";};
    //@}
};

#endif
