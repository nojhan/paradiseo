// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTerm.h
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

#ifndef _EOTERM_H
#define _EOTERM_H

#include <eoPop.h>

/** Termination condition for the genetic algorithm
 * Takes the population as input, returns true for continue,
 * false for termination
 */
template< class EOT>
class eoTerm {
public:

	/// Ctors/dtors
	virtual ~eoTerm() {};

	/** Returns false if the training has to stop, true if it
	 continues \\
	 It is non-const since it might change the internal state
	 of the object, for instance, updating a counter
	*/
	virtual bool operator() ( const eoPop< EOT >& _pop ) = 0 ;
	
};

#endif
