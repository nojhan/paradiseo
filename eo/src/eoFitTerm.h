// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGenTerm.h
// (c) GeNeura Team, 1999
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

#ifndef _EOFITTERM_H
#define _EOFITTERM_H

#include <eoTerm.h>


/** Fitness termination: terminates after a the difference between the
fitness of the best individual and a maximum fitness to achieve is less
than certain number called epsilon., i.e., |maximum-fitness|<epsilon
*/
template< class EOT>
class eoFitTerm: public eoTerm<EOT> {
public:

	/// Ctors/dtors
	eoFitTerm( const float _maximum, const float _epsilon )
		: eoTerm<EOT> (), maximum( _maximum ), epsilon(_epsilon){};

	/// Copy ctor
	eoFitTerm( const eoFitTerm&  _t )
		: eoTerm<EOT> ( _t ), maximum( _t.maximum ), 
	  epsilon(_t.epsilon){};

	///
	virtual ~eoFitTerm() {};

	/** Returns false when a fitness criterium is
	* reached */
	virtual bool operator() ( const eoPop<EOT>& _vEO ) {
	  float bestFitness=_vEO[0].fitness();
	  float dif=bestFitness-maximum;
	  dif=(dif<0)?-dif:dif;
	  return (dif>epsilon ) || (bestFitness > maximum);
	}

	std::string className(void) const { return "eoFitTerm"; }

private:
	float maximum, epsilon;
};

#endif

