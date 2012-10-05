// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoString.h
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

#ifndef _eoString_H
#define _eoString_H

// STL libraries
#include <iostream>
#include <string>
#include <stdexcept>


#include <EO.h>

//-----------------------------------------------------------------------------
// eoString
//-----------------------------------------------------------------------------

/** Adaptor that turns an STL std::string into an EO

  @ingroup Representations
  @ingroup Utilities
 */
template <class fitnessT >
class eoString: public EO<fitnessT>, public std::string
{
public:

    typedef char Type;
    typedef char AtomType;
    typedef std::string   ContainerType;


  /// Canonical part of the objects: several ctors, copy ctor, dtor and assignment operator
  //@{
  /// ctor
  eoString( const std::string& _str ="" )
    : std::string( _str ) {};

  /// printing...
  virtual void printOn(std::ostream& os) const
  {
    EO<fitnessT>::printOn(os);
    os << ' ';

    os << size() << ' ' << substr() << std::endl;

  }

  /** @name Methods from eoObject
      readFrom and printOn are directly inherited from eo1d
  */
  //@{
  /** Inherited from eoObject
      @see eoObject
  */
  virtual std::string className() const {return "eoString";};
  //@}


};

#endif
