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

#ifndef _EOGENTERM_H
#define _EOGENTERM_H

#include <eoTerm.h>

/** Generational termination: terminates after a number of generations
*/
template< class EOT>
class eoGenTerm: public eoTerm<EOT> {
public:

	/// Ctors/dtors
	eoGenTerm( unsigned _totalGens)
	  : eoTerm<EOT> (), repTotalGenerations( _totalGens ), 
	  thisGeneration(0){};

	/// Copy Ctor
	eoGenTerm( const eoGenTerm& _t)
	  : eoTerm<EOT> ( _t ), repTotalGenerations( _t.repTotalGenerations ), 
	  thisGeneration(0){};

	/// Assignment Operator
	const eoGenTerm& operator = ( const eoGenTerm& _t) {
	  if ( &_t != this ) {
	    repTotalGenerations =  _t.repTotalGenerations; 
	    thisGeneration = 0;
	  }
	  return *this;
	}

	/// Dtor
	virtual ~eoGenTerm() {};

	/** Returns false when a certain number of generations is
	* reached */
	virtual bool operator() ( const eoPop<EOT>& _vEO ) {
	  thisGeneration++;
	  //	  cout << " [" << thisGeneration << "] ";
	  return (thisGeneration < repTotalGenerations) ; // for the postincrement
	}

	/** Sets the number of generations to reach 
	    and sets the current generation to 0 (the begin)*/
	virtual void totalGenerations( unsigned _tg ) { 
	  repTotalGenerations = _tg; 
	  //	  thisGeneration = 0;
	};

	/** Returns the number of generations to reach*/
	virtual unsigned totalGenerations( ) {  
	  return repTotalGenerations; 
	};
	
private:
	unsigned repTotalGenerations, thisGeneration;
};

#endif
