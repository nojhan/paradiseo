// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoXOver2.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EOXOVER2_h
#define _EOXOVER2_h


// for swap
#if defined( __BORLANDC__ )
#include <algorith>
#else
#include <algorithm>
#endif

// EO includes
#include <eoOp.h>
#include <eoUniform.h>

/** 2-point crossover: takes the genes in the central section of two EOs
and interchanges it
*/
template <class EOT>
class eoXOver2: public eoBinOp<EOT> {
public:
	///
	eoXOver2() 
	  : eoBinOp< EOT >(){};

	///
	virtual ~eoXOver2() {};

	///
	virtual void operator()( EOT& _eo1, 
							EOT& _eo2 ) const {
	  unsigned len1 = _eo1.length(), len2 = _eo2.length(), 
	    len= (len1 > len2)?len2:len1;
	  eoUniform<unsigned> uniform( 0, len );
	  unsigned pos1 = uniform(), pos2 = uniform() ;
	  
	  applyAt( _eo1, _eo2, pos1, pos2 );
	  
	}

	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eoOp
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	string className() const {return "eoXOver2";};
    //@}

private:

#ifdef _MSC_VER
	typedef EOT::Type Type;
#else
	typedef typename EOT::Type Type;
#endif

	/// applies operator to one gene in the EO
	virtual void applyAt( EOT& _eo, EOT& _eo2, 
			      unsigned _i, unsigned _j = 0) const {
	  
	  if ( _j < _i )
	    swap( _i, _j );

	  unsigned len1 = _eo.length(), len2 = _eo2.length(), 
	    len= (len1 > len2)?len2:len1;

	  if ( (_j > len) || (_i> len ) ) 
	    throw runtime_error( "xOver2: applying xOver past boundaries");
	  
	  for (  unsigned i = _i; i < _j; i++ ) {
	    Type tmp = _eo.gene( i );
	    _eo.gene( i ) = _eo2.gene( i );
	    _eo2.gene( i ) = tmp ;
	  }
	  
	}

};

#endif
