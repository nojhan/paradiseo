// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEvaluator.h
// (c) GeNeura Team, 1998
/* 
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */
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
