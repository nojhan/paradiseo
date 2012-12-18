// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFlOrQuadOp.h
// (c) Marc Schoenauer - Maarten Keijzer 2000-2003
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
             mkeijzer@cs.vu.nl
 */
//-----------------------------------------------------------------------------

#ifndef _eoFlOrQuadOp_h
#define _eoFlOrQuadOp_h

#include <eoFunctor.h>
#include <eoOp.h>

/** @addtogroup Variators
 * @{
 */

/** Generic eoQuadOps on fixed length genotypes.
 *  Contains exchange crossovers (1pt and uniform)
 *      and crossovers that applies an Atom crossover
 *          to all components with given rate, or
 *          to a fixed prescribed nb of components
*/

//////////////////////////////////////////////////////////////////////
//                        eoFlOrAllAtomQuadOp
//////////////////////////////////////////////////////////////////////

/** Quad Crossover using an Atom Crossover
 */
template <class EOT>
class eoFlOrAllAtomQuadOp : public eoQuadOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;

  /** default ctor: requires an Atom QuadOp */
  eoFlOrAllAtomQuadOp( eoQuadOp<AtomType>& _op, double _rate = 1):
    op(_op), rate( _rate ) {}

  /** applies Atom crossover to ALL components with given rate */
  bool operator()(EOT & _eo1, EOT & _eo2)
  {
    bool changed = false;
    for ( unsigned i = 0; i < _eo1.size(); i++ ) {
      if ( rng.flip( rate ) ) {
        bool changedHere = op( _eo1[i], _eo2[i] );
        changed |= changedHere;
      }
    }
    return changed;
  }

  /** inherited className()*/
  virtual string className() const { return "eoFlOrAllAtomQuadOp"; }

private:
  double rate;
  eoQuadOp<AtomType> & op;
};

//////////////////////////////////////////////////////////////////////
//                        eoFlOrKAtomQuadOp
//////////////////////////////////////////////////////////////////////
/** Quad Crossover using an Atom Crossover
 *  that is applied to a FIXED NB of components
 */
template <class EOT>
class eoFlOrKAtomQuadOp : public eoQuadOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;

  /** default ctor: requires an Atom QuadOp and an unsigned */
  eoFlOrAtomQuadOp( eoQuadOp<AtomType>& _op, unsigned _k = 1):
    op(_op), k( _k ) {}

  /** applies the Atom QuadOp to some components */
  bool operator()(EOT & _eo1, const EOT & _eo2)
  {
    if (_eo1.size() != _eo2.size())
      {
        string s = "Operand size don't match in " + className();
        throw runtime_error(s);
      }

    bool changed = false;
    for ( unsigned i = 0; i < k; i++ ) //! @todo check that we don't do twice the same
      {
        unsigned where = eo::rng.random(_eo1.size());
        bool changedHere = op( _eo1[where], _eo2[where] );
        changed |= changedHere;
      }
    return changed;
  }

  /** inherited className()*/
  virtual string className() const { return "eoFlOrKAtomQuadOp"; }

private:
  unsigned k;
  eoQuadOp<AtomType> & op;
};


//////////////////////////////////////////////////////////////////////
//                        eoFlOrUniformQuadOp
//////////////////////////////////////////////////////////////////////
/** The uniform crossover - exchanges atoms uniformly ! */
template <class EOT>
class eoFlOrUniformQuadOp : public eoQuadOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;

  /** default ctor: requires a rate - 0.5 by default */
  eoVlUniformQuadOp(double _rate=0.5) : eoQuadOp<EOT>(_size),
    rate(_rate) {}

  /** excahnges atoms at given rate */
  bool operator()(EOT & _eo1, EOT & _eo2)
  {
    unsigned i;
    Atom tmp;
    if (_eo1.size() != _eo2.size())
      {
        string s = "Operand size don't match in " + className();
        throw runtime_error(s);
  }
    bool hasChanged = false;
    for (unsigned i=0; i<_eo1.size(); i++)
      {
        if ( (_eo1[i]!=_eo2[i]) && (eo::rng.filp(rate)) )
        {
          tmp = _eo1[i];
          _eo1[i] = _eo2[i];
          _eo2[i] = tmp;
          hasChanged = true;
        }
      }
    return hasChanged;
  }

  /** inherited className()*/
  virtual string className() const { return "eoFlOrUniformQuadOp"; }

private:
  double rate;
};

//////////////////////////////////////////////////////////////////////
//                        eoFlOr1ptQuadOp
//////////////////////////////////////////////////////////////////////
/** The 1pt  crossover (just in case someone wants it some day!) */
template <class EOT>
class eoFlOr1ptQuadOp : public eoQuadOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;

  /** default ctor: no argument */
  eoVlUniformQuadOp() {}

  /** exchanges first and second parts of the vectors of Atoms */
  bool operator()(EOT & _eo1, EOT & _eo2)
  {
    unsigned i;
    Atom tmp;
    if (_eo1.size() != _eo2.size())
      {
        string s = "Operand size don't match in " + className();
        throw runtime_error(s);
  }
    bool hasChanged = false;
    unsigned where = eo::rng.random(_eo1.size()-1);
    for (unsigned i=where+1; i<_eo1.size(); i++)
      {
        if ( (_eo1[i]!=_eo2[i]) )
        {
          tmp = _eo1[i];
          _eo1[i] = _eo2[i];
          _eo2[i] = tmp;
          hasChanged = true;
        }
      }
    return hasChanged;
  }

  /** inherited className()*/
  virtual string className() const { return "eoFlOr1ptQuadOp"; }

};

/** @} */

#endif
