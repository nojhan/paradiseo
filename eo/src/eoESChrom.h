// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoESChrom.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------


#ifndef _eoESCHROM_H
#define _eoESCHROM_H

// STL libraries
#include <vector>		// For vector<>
#include <stdexcept>
#include <strstream>
#include <iostream>		// for ostream

// EO includes
#include <eoVector.h>

/**@name Chromosomes for evolution strategies
Each chromosome in an evolution strategies is composed of a vector of floating point
values plus a vector of sigmas, that are added to them during mutation
*/ 
//@{

/** Each gene in an Evolution Strategies is composed of a value plus an standard
deviation, sigma, used for mutation*/
struct eoESGene {
	double val, sigma;
	eoESGene( double _val = 0, double _sigma = 0 ): val( _val ), sigma( _sigma ) {};
};

/// Tricky operator to avoid errors in some VC++ systems, namely VC 5.0 SP3
bool operator < ( eoESGene _e1, eoESGene _e2 ) {
	return _e1.val < _e2.val;
}

/// Tricky operator to avoid errors in some VC++ systems
bool operator == ( eoESGene _e1, eoESGene _e2 ) {
	return ( _e1.val == _e2.val ) && ( _e1.sigma == _e2.sigma ) ;
}

///
ostream & operator << ( ostream& _s, const eoESGene& _e ) {
	_s << _e.val << ", " << _e.sigma << " | ";
	return _s;
}

/// Dummy >>
istream & operator >> ( istream& _s, const eoESGene& _e ) {
	_s >> _e.val;
	_s >> _e.sigma;
	return _s;
}


/** Basic chromosome for evolution strategies (ES), as defined by Rechenberg and
Schwefel. Each chromosomes is composed of "genes"; each one of then is an eoESGene
@see eoESGene
*/
template <typename fitT = float >
class eoESChrom: public eoVector<eoESGene, fitT> {	
public:
	/// Basic ctor
	eoESChrom( ):eoVector<eoESGene, fitT>() {};

	/** Ctor using a couple of random number generators
	@param _size Lineal length of the object
	@param _rnd a random number generator, which returns a random value each time it´s called.
	@param _rndS another one, for the sigma
	*/
	eoESChrom( unsigned _size, eoRnd<double>& _rnd, eoRnd<double>& _rndS  )
		: eoVector<eoESGene, fitT>( _size ){ 
		for ( iterator i = begin(); i != end(); i ++ ) {
			i->val = _rnd();
			i->sigma = _rndS();
		}
	};

	/// Copy ctor
	eoESChrom( const eoESChrom& _eoes): eoVector<eoESGene, fitT>( _eoes ) {};

	/// Assignment operator
	const eoESChrom& operator =( const eoESChrom & _eoes ) {
		if ( this != &_eoes ){
			eoVector<eoESGene, fitT>::operator=( _eoes );
		}
		return *this;
	}

	///
	~eoESChrom() {};

	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eo1d
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	string className() const {return "eoESChrom";};
    //@}

};

//@}
#endif
