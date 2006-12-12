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

#include <eoContinue.h>


/** 
Fitness continuation: 

  Continues until the maximum fitness level is reached.
*/
template< class EOT>
class eoFitContinue: public eoContinue<EOT> {
public:

    /// Define Fitness
    typedef typename EOT::Fitness FitnessType;

	/// Ctor
    eoFitContinue( const FitnessType _maximum)
		: eoContinuator<EOT> (), maximum( _maximum ) {};

	/** Returns false when a fitness criterium is
	* reached, assumes sorted population */
	virtual bool operator() ( const eoPop<EOT>& _vEO ) 
    {
	  return (bestFitness < maximum);
	}

private:
	FitnessType maximum;
};

#endif

