// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEvaluator.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EOEVALUATOR_H
#define _EOEVALUATOR_H

//-----------------------------------------------------------------------------
#include <eoPopOps.h>
#include <eoEvalFunc.h>

//-----------------------------------------------------------------------------
/** Evaluator takes a vector of EOs and evaluates its fitness
* returning void. Template instances must be of fitness and EO type
*/
template<class EOT>
class eoEvaluator: public eoTransform<EOT>{
public:
	/// ctor
	eoEvaluator( const eoEvalFunc< EOT> & _ef )
		: eoTransform<EOT>(), repEF( _ef ){};

	/// Needed virtual destructor
	virtual ~eoEvaluator() {};

	/* Sets the evaluation function
	virtual void EF( const eoEvalFunc< EOT> & _ef ) { repEF= _ef;};*/

	/// Gets the evaluation function
	virtual const eoEvalFunc< EOT>& EF() { return repEF;};

	/** This is the actual function operator(); it is left without implementation.
		It takes a vector of pointers to eo
	 * @param _vEO is a vector of pointers to eo, that will be evaluated
	 */

	///@name eoObject methods
	//@{
	/** Return the class id */
	virtual string className() const { return "eoEvaluator"; }

	/** Read and print are left without implementation */
	//@}
	
private:
	const eoEvalFunc< EOT> & repEF; 
};
//@}

#endif
