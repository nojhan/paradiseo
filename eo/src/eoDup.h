// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoKill.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EODUP_h
#define _EODUP_h

#include <eoUniform.h>

#include <eoOp.h>

/// Dup or duplicate: duplicates a gene in a chromosome
template <class EOT>
class eoDup: public eoMonOp<EOT>  {
public:
	///
	eoDup( )
		: eoMonOp< EOT >( ){};

	/// needed virtual dtor
	virtual ~eoDup() {};

	///
	virtual void operator()( EOT& _eo ) const {
		eoUniform<unsigned> uniform( 0, _eo.length() );
		unsigned pos = uniform();
		_eo.insertGene( pos, _eo.gene(pos) );
	}


	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eoOp
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	virtual string className() const {return "eoDup";};
    //@}
};

#endif
