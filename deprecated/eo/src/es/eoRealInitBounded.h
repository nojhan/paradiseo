// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRealInitBounded.h
// (c) EEAAX 2000 - Maarten Keijzer 2000
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

    Contact: Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoRealInitBounded_h
#define eoRealInitBounded_h

//-----------------------------------------------------------------------------

#include <utils/eoRNG.h>
#include <eoInit.h>
#include <es/eoReal.h>
#include <utils/eoRealVectorBounds.h>

/** Simple initialization for any EOT that derives from std::vector<double>
 *    uniformly in some bounds
 *
 * @ingroup Real
 * @ingroup Variators
 */
template <class EOT>
class eoRealInitBounded : public eoInit<EOT>
{
 public:
  /** Ctor - from eoRealVectorBounds */
  eoRealInitBounded(eoRealVectorBounds & _bounds):bounds(_bounds)
  {
        if (!bounds.isBounded())
      throw std::runtime_error("Needs bounded bounds to initialize a std::vector<double>");
  }

  /** simply passes the argument to the uniform method of the bounds */
  virtual void operator()(EOT & _eo)
    {
      bounds.uniform(_eo);  // resizes, and fills uniformly in bounds
      _eo.invalidate();            // was MISSING!!!!
    }

  /** accessor to the bounds */
  virtual eoRealVectorBounds & theBounds() {return bounds;}
  virtual unsigned size(){return bounds.size();}

 private:
  eoRealVectorBounds & bounds;
};

//-----------------------------------------------------------------------------
//@}
#endif
