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
#include <utils/eoRealBounds.h>

template <class FitT>
class eoRealInitBounded : public eoInit<eoReal<FitT> >
{
 public:
  /** Ctor - from eoRealVectorBounds */
  eoRealInitBounded(eoRealVectorBounds & _bounds):bounds(_bounds) {}

  /** simply passes the argument to the uniform method of the bounds */
  virtual void operator()(eoReal<FitT>& _eo)
    {
      bounds.uniform(_eo);	   // fills _eo with uniform values in bounds
    }

  /** accessor to the bounds */
  const eoRealVectorBounds & theBounds() {return bounds;}

 private:
  eoRealVectorBounds & bounds;
};

//-----------------------------------------------------------------------------
//@}
#endif eoRealOp_h

