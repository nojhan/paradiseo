// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoCombinedContinue.h
// (c) Maarten Keijzer, GeNeura Team, 1999, 2000
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

#ifndef _eoCombinedContinue_h
#define _eoCombinedContinue_h

#include <eoContinue.h>

/** 
Fitness continuation: 

  Continues until one of the embedded continuators says halt!
*/
template< class EOT>
class eoCombinedContinue: public eoContinue<EOT> {
public:

    /// Define Fitness
    typedef typename EOT::Fitness FitnessType;

	/// Ctor
    eoCombinedContinue( eoContinue<EOT>& _arg1, eoContinue<EOT>& _arg2)
		: eoContinue<EOT> (), arg1(_arg1), arg2(_arg2) {};

	/** Returns false when one of the embedded continuators say so (logical and)
	*/
	virtual bool operator() ( const eoPop<EOT>& _pop ) 
    {
        return arg1(_pop) && arg2(_pop);
	}

private:
    eoContinue<EOT>& arg1;
    eoContinue<EOT>& arg2;
};

#endif

