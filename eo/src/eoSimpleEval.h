// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSimpleEval.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EOSimpleEval_H
#define _EOSimpleEval_H

#include <eoEvaluator.h>	// for evalFunc
#include <algorithm>		// For sort

/** Particular instantiation of the EOEvaluator class
It takes each member in the population, and evaluates it, applying
the evaluation function it´s been initialized with 
*/
template<class EOT>
class eoSimpleEval: public eoEvaluator<EOT> {
public:

	/// Ctors/dtors
	eoSimpleEval( eoEvalFunc<EOT> & _ef ):eoEvaluator<EOT>(_ef) {};
	
	///
	virtual ~eoSimpleEval() {};

#ifdef _MSC_VER
	typedef EOT::Fitness EOFitT;
#else
	typedef typename EOT::Fitness EOFitT;
#endif
	/** Applies evaluation function to all members in the population, and sets
	    their fitnesses
	Reference is non-const since it orders the population by any order 
	it´s been defined 
	@param _vEO the population whose fitness is going to be computed*/
	virtual void operator() ( eoPop< EOT >& _vEO ) {
	  for ( eoPop<EOT>::iterator i = _vEO.begin(); i != _vEO.end(); i ++ ){
	    i->fitness( EF().evaluate( *i ) );
	  }
	  sort( _vEO.begin(), _vEO.end() ); 
	};

		///@name eoObject methods
	//@{
	///
	void printOn( ostream& _os ) const {};
	///
	void readFrom( istream& _is ){};

	///
	string className() { return "eoSimpleEval";};

	//@}
	
};

#endif
