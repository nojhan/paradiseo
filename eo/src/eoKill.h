// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoKill.h
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

#ifndef _EOKILL_h
#define _EOKILL_h

#include <eoUniform.h>

#include <eoOp.h>

/// Kill eliminates a gen in a chromosome
template <class EOT >
class eoKill: public eoMonOp<EOT>  {
public:
	///
	eoKill( )
		: eoMonOp< EOT >(){};

	/// needed virtual dtor
	virtual ~eoKill() {};

	///
	virtual void operator()( EOT& _eo ) const {
	  eoUniform<unsigned> uniform( 0, _eo.length() );
	  unsigned pos = uniform( );
	  _eo.deleteGene( pos );
	}

	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eoOp
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	virtual string className() const {return "eoKill";};
    //@}
};

#endif
