/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    -----------------------------------------------------------------------------
    eoObject.h
      This is the base class for most objects in EO. It basically defines an interf
    face for giving names to classes.

    (c) GeNeura Team, 1998, 1999, 2000
 
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

#ifndef EODISTANCE_H
#define EODISTANCE_H

//-----------------------------------------------------------------------------

using namespace std;

//-----------------------------------------------------------------------------
// eoDistance
//-----------------------------------------------------------------------------
/** Defines an interface for measuring distances between evolving objects */
template <class EOT>
class eoDistance {
 public:
  
  /// Default Constructor.
  eoDistance() {}

  /// Copy constructor.
  eoDistance( const eoDistance& ) {}

  /// Virtual dtor. They are needed in virtual class hierarchies.
  virtual ~eoDistance() {}
  

  /** Return the distance from the object with this interface to other
      object of the same type.
  */
  virtual double distance( const EOT& ) const = 0;

  /// Returns classname
  virtual std::string className() const { return "eoDistance"; }

};

#endif EOOBJECT_H
