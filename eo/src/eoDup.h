// eoDup.h
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

#ifndef _EODUP_h
#define _EODUP_h

#include <utils/eoRNG.h>

#include <eoOp.h>

/// Dup or duplicate: duplicates a gene in a chromosome
template <class EOT>
class eoDup: public eoMonOp<EOT>  {
public:
	///
	eoDup( )
		: eoMonOp< EOT >( ){};

	/// needed virtual dtor
	virtual ~eoDup() {};

	///
	virtual void operator()( EOT& _eo ) const 
    {
		unsigned pos = rng.random(_eo.length());
		_eo.insertGene( pos, _eo.gene(pos) );
	}


	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eoOp
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	virtual string className() const {return "eoDup";};
    //@}
};

#endif

