// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoMultiMonOp.h
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

#ifndef _EOMULTIMONOP_h
#define _EOMULTIMONOP_h

#include <iterator>

#include <eoOp.h>

/** MultiMonOp combines several monary operators. By itself, it does nothing to the
EO it´s handled*/
template <class EOT>
class eoMultiMonOp: public eoMonOp<EOT>  {
public:
	/// Ctor from an already existing op
	eoMultiMonOp( const eoMonOp<EOT>* _op )
		: eoMonOp< EOT >( ), vOp(){
		vOp.push_back( _op );
	};

	///
	eoMultiMonOp( )
		: eoMonOp< EOT >( ), vOp(){};

	/// Ctor from an already existing op
	void adOp( const eoMonOp<EOT>* _op ){
		vOp.push_back( _op );
	};

	/// needed virtual dtor
	virtual ~eoMultiMonOp() {};

	///
	/// Applies all operators to the EO
	virtual void operator()( EOT& _eo ) const {
		if ( vOp.begin() != vOp.end() ) {
			for ( vector<const eoMonOp<EOT>* >::const_iterator i = vOp.begin(); 
					i != vOp.end(); i++ ) {
				(*i)->operator () ( _eo );
			}
		}
	}


	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eoOp
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	string className() const {return "eoMonOp";};
    //@}
private:
	vector< const eoMonOp<EOT>* > vOp;
};

#endif