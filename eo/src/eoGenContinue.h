// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGenContinue.h
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

#ifndef _eoGenContinue_h
#define _eoGenContinue_h

#include <eoContinue.h>

/** 
    Generational continuator: continues until a number of generations is reached
*/
template< class EOT>
class eoGenContinue: public eoContinue<EOT>
{
public:

	/// Ctor for setting a
	eoGenContinue( unsigned _totalGens)
	  : repTotalGenerations( _totalGens ), 
      thisGenerationPlaceHolder(0),
	  thisGeneration(thisGenerationPlaceHolder){};
	
    /// Ctor for enabling the save/load the no. of generations counted
	eoGenContinue( unsigned _totalGens, unsigned& _currentGen)
	  : repTotalGenerations( _totalGens ), 
      thisGenerationPlaceHolder(0),
	  thisGeneration(_currentGen){};

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
	  thisGeneration = 0;
	};

	/** Returns the number of generations to reach*/
	virtual unsigned totalGenerations( ) 
    {  
	  return repTotalGenerations; 
	};
    
private:
	unsigned repTotalGenerations;
    unsigned thisGenerationPlaceHolder;
    unsigned& thisGeneration;
};

#endif

