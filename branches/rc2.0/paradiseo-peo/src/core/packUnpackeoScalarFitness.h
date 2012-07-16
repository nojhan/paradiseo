/*
 C++ Interface: packUnpackeoScalarFitness

(c) TAO Project Team of INRIA Saclay, 2010 Thales group

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; version 2
    of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
    Mostepha-Redouane Khouadjia <mostepha-redouane.khouadjia@inria.fr>
    Johann Dréo <johann.dreo@thalesgroup.com>
*/
#ifndef _packUnpackeoScalarFitness_h
#define _packUnpackeoScalarFitness_h

#include "messaging.h"

#include <utility>
#include <string>


template <class ScalarType >
void pack( const  eoScalarFitness  < ScalarType, std::less<ScalarType> >   & __fit ) {

	ScalarType  value  = __fit;  // pack the scalar type of the fitness ( see  operator ScalarType () in <eo> )
	
	pack( value ); 
	
	}
	
	
	
	
template <class ScalarType >
void pack( const  eoScalarFitness  < ScalarType, std::greater<ScalarType> >   & __fit ) {

	ScalarType  value  = __fit; // same as for less<ScalarType>
	
	pack( value ); 
	
	}
	
	
template <class ScalarType >
void unpack(  eoScalarFitness  < ScalarType, std::less<ScalarType> > & __fit ) {

	ScalarType  value ;
	
	unpack( value ); 
	
	}
	
	
	
		
template <class ScalarType >
void unpack(  eoScalarFitness  < ScalarType, std::greater<ScalarType> > & __fit ) {

	ScalarType  value ;
	
	unpack( value ); 
	
	}
	
	
#endif	
	
